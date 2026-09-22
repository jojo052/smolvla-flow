"""Official T2 adapter. No student overlay; native two-step RTC sampling."""
import hashlib
import inspect
import math
import time


class Policy:
    def __init__(self,args):
        import torch
        from scripts.run_libero_rollout import _load_policy,parse_args
        from smolvla_flow.async_runtime import LeRobotPostprocessorAdapter
        self.torch=torch
        self.loaded=_load_policy(parse_args(['--checkpoint',str(args.checkpoint),'--flow-steps','2',
            '--mode','async','--rtc-max-guidance-weight','10','--rtc-schedule','exp']),action_execution_steps=10)
        self.policy=self.loaded.policy
        self.policy.eval()
        for p in self.policy.parameters():p.requires_grad_(False)
        self.model=self.policy.model
        self.rtc=getattr(self.model,'rtc_processor',None)
        if self.rtc is None:raise RuntimeError('Locked model does not expose RTCProcessor')
        self.rtc.rtc_config.max_guidance_weight=1.25
        self.post=LeRobotPostprocessorAdapter(self.loaded.postprocessor)
        self.original_denoise=self.model.denoise_step
        self.original_rtc=self.rtc.denoise_step
        self.model.denoise_step=self.counted_denoise
        self.rtc.denoise_step=self.counted_rtc
        self.prepares={};self.reset()

    def reset(self):
        self.generator=self.torch.Generator(device='cuda').manual_seed(123)
        self.policy.reset()

    def counted_denoise(self,*args,**kwargs):
        self.forward_calls+=1
        a=self.torch.cuda.Event(enable_timing=True);b=self.torch.cuda.Event(enable_timing=True)
        a.record()
        if self.diagnostic:
            bound=inspect.signature(self.original_denoise).bind(*args,**kwargs)
            self.times.append(bound.arguments['timestep'].detach().cpu().tolist())
        value=self.original_denoise(*args,**kwargs)
        b.record();self.forward_events.append((a,b))
        return value

    def counted_rtc(self,*args,**kwargs):
        self.rtc_calls+=1
        a=self.torch.cuda.Event(enable_timing=True);b=self.torch.cuda.Event(enable_timing=True)
        a.record()
        if self.diagnostic:
            bound=inspect.signature(self.original_rtc).bind(*args,**kwargs)
            original=bound.arguments['original_denoise_step_partial'];base=[]
            def track(*aa,**kk):
                value=original(*aa,**kk);base.append(value.detach().clone());return value
            bound.arguments['original_denoise_step_partial']=track
            out=self.original_rtc(*bound.args,**bound.kwargs)
            if not base:raise RuntimeError('RTC never evaluated native velocity')
            self.corrections.append(float((out.detach()-base[-1]).abs().max()))
        else:out=self.original_rtc(*args,**kwargs)
        b.record();self.rtc_events.append((a,b))
        return out

    def predict(self,obs,language,mode,delay_ms,prefix=None,estimated_delay=0,noise=None,diagnostic=False):
        from scripts.run_libero_rollout import _make_observation_pipeline
        torch=self.torch;self.diagnostic=diagnostic
        self.forward_calls=0;self.rtc_calls=0;self.times=[];self.corrections=[];self.rtc_events=[];self.forward_events=[]
        start=time.perf_counter()
        enabled=mode=='rtc' and bool(prefix)
        self.model.rtc_processor=self.rtc if enabled else None
        for cfg in (self.policy.config,self.model.config):
            if cfg.rtc_config is not None:cfg.rtc_config.enabled=enabled
        if language not in self.prepares:self.prepares[language]=_make_observation_pipeline(language)
        batch=self.prepares[language](obs,self.loaded.preprocessor)
        if tuple(batch['observation.state'].shape)!=(1,8):raise ValueError('State contract')
        images=[x for k,x in batch.items() if k.startswith('observation.images.')]
        # The input pipeline is 256px; policy.prepare_images performs internal resizing.
        if len(images)!=2 or any(tuple(x.shape)!=(1,3,256,256) for x in images):raise ValueError('Input image contract')
        if noise is None:noise=torch.normal(0.,1.,size=(1,50,32),device='cuda',generator=self.generator)
        noise_hash=hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest()
        previous=torch.tensor(prefix,device='cuda',dtype=torch.float32).unsqueeze(0) if enabled else None
        torch.cuda.synchronize();compute=time.perf_counter()
        with torch.enable_grad():
            chunk=self.policy._get_action_chunk(batch,noise=noise,
                prev_chunk_left_over=previous,inference_delay=estimated_delay if enabled else 0,execution_horizon=10)
        torch.cuda.synchronize();prediction_s=time.perf_counter()-compute
        if self.forward_calls!=2 or self.rtc_calls!=(2 if enabled else 0):
            raise ValueError('Two-step/RTC execution path drift')
        if tuple(chunk.shape)!=(1,50,7) or not torch.isfinite(chunk).all():raise ValueError('Invalid output chunk')
        normalized=chunk[0].detach();actions=[self.post(a).tolist() for a in normalized]
        ready_compute=time.perf_counter()
        if delay_ms:time.sleep(delay_ms/1000)
        done=time.perf_counter()
        if any(p.grad is not None for p in self.policy.parameters()):raise RuntimeError('Unexpected parameter gradient')
        forward_s=sum(a.elapsed_time(b) for a,b in self.forward_events)/1000
        rtc_s=sum(a.elapsed_time(b) for a,b in self.rtc_events)/1000
        return dict(normalized=normalized.cpu().tolist(),actions=actions,
            metrics=dict(prediction_s=prediction_s,preprocess_s=compute-start,
                postprocess_s=ready_compute-compute-prediction_s,injected_wait_s=done-ready_compute,
                elapsed_s=done-start,forward_calls=self.forward_calls,rtc_calls=self.rtc_calls,
                rtc_processor_inclusive_s=rtc_s,denoise_forward_s=forward_s,
                rtc_extra_s=max(0.,rtc_s-forward_s) if enabled else 0.,
                noise_sha256=noise_hash,timesteps=self.times,correction_max=self.corrections,
                estimated_delay=estimated_delay,rtc_active=enabled))

    def digest(self):
        h=hashlib.sha256()
        for name,p in self.policy.named_parameters():
            h.update(name.encode());h.update(p.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def preflight(self,obs,language):
        before=self.digest();results=[]
        for seed in (123,124):
            noise=self.torch.normal(0.,1.,size=(1,50,32),device='cuda',
                generator=self.torch.Generator(device='cuda').manual_seed(seed))
            plain=self.predict(obs,language,'async',0,noise=noise.clone(),diagnostic=True)
            prefix=[[x+.2 for x in a] for a in plain['normalized'][10:]]
            guided=self.predict(obs,language,'rtc',0,prefix,2,noise.clone(),diagnostic=True)
            for value in (plain,guided):
                if value['metrics']['timesteps']!=[[1.0],[0.5]]:
                    raise ValueError('Expected t=1,0.5; dt=-0.5')
            if plain['metrics']['forward_calls']!=2 or guided['metrics']['rtc_calls']!=2:
                raise ValueError('Expected two Euler denoising steps')
            delta=max(abs(a-b) for aa,bb in zip(plain['normalized'],guided['normalized']) for a,b in zip(aa,bb))
            correction=max(guided['metrics']['correction_max'],default=0.)
            results.append(dict(seed=seed,output_max_difference=delta,guidance_max=correction,
                                plain=plain['metrics'],guided=guided['metrics']))
        if self.digest()!=before:raise ValueError('Model parameters changed')
        passed=all(math.isfinite(x['guidance_max']) for x in results) and any(
            x['guidance_max']>1e-7 and x['output_max_difference']>1e-7 for x in results)
        return dict(passed=passed,checks=results,parameter_sha256=before,
                    reason=None if passed else 'Two-step RTC guidance is ineffective')
