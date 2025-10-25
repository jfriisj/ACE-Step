"""
Optimized FastAPI server for ACE-Step with performance enhancements
Integrates model caching + Phase 1 optimizations for maximum performance
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import uuid
import time
import asyncio
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from acestep.pipeline_ace_step import ACEStepPipeline

app = FastAPI(title="ACEStep Optimized API", version="2.0.0")

# Global optimized model cache  
_model_cache = {}
_performance_stats = {
    "requests_processed": 0,
    "total_inference_time": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
    "avg_response_time": 0.0
}

# Threading for parallel preprocessing
_preprocessing_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ace_preprocess")

class ACEStepInput(BaseModel):
    checkpoint_path: str
    bf16: bool = True
    torch_compile: bool = False
    device_id: int = 0
    output_path: Optional[str] = None
    audio_duration: float
    prompt: str
    lyrics: str
    infer_step: int
    guidance_scale: float
    scheduler_type: str
    cfg_type: str
    omega_scale: float
    actual_seeds: List[int]
    guidance_interval: float
    guidance_interval_decay: float
    min_guidance_scale: float
    use_erg_tag: bool
    use_erg_lyric: bool
    use_erg_diffusion: bool
    oss_steps: List[int]
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0
    
    # Optimization parameters
    enable_optimizations: bool = True
    quick_warmup: bool = True
    parallel_preprocessing: bool = True

class ACEStepOutput(BaseModel):
    status: str
    output_path: Optional[str]
    message: str
    performance_metrics: Optional[Dict[str, Any]] = None

class PerformanceStats(BaseModel):
    requests_processed: int
    total_inference_time: float
    cache_hits: int
    cache_misses: int
    avg_response_time: float
    cached_models: int
    cache_keys: List[str]

def get_or_create_pipeline(
    checkpoint_path: str, 
    bf16: bool, 
    torch_compile: bool, 
    device_id: int,
    enable_optimizations: bool = True
) -> ACEStepPipeline:
    """
    Get cached optimized pipeline or create new one if not exists
    """
    global _performance_stats
    
    # Create cache key from parameters
    cache_key = f"{checkpoint_path}_{bf16}_{torch_compile}_{device_id}_{enable_optimizations}"
    
    if cache_key not in _model_cache:
        logger.info(f"🚀 Creating new optimized pipeline for cache key: {cache_key}")
        _performance_stats["cache_misses"] += 1
        
        start_time = time.time()
        
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        # Create pipeline
        pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            dtype="bfloat16" if bf16 else "float32",
            torch_compile=torch_compile,
        )
        
        # Add optimization attributes to existing pipeline
        if enable_optimizations:
            _enhance_pipeline_with_optimizations(pipeline)
        
        creation_time = time.time() - start_time
        
        _model_cache[cache_key] = pipeline
        logger.info(f"✅ Optimized pipeline cached with key: {cache_key} (created in {creation_time:.2f}s)")
        
    else:
        logger.info(f"⚡ Using cached optimized pipeline for key: {cache_key}")
        _performance_stats["cache_hits"] += 1
    
    return _model_cache[cache_key]

def _enhance_pipeline_with_optimizations(pipeline: ACEStepPipeline):
    """Add optimization methods to existing pipeline"""
    import torch
    import gc
    
    # Add performance tracking
    pipeline._performance_metrics = {
        'warmup_time': 0,
        'preprocessing_time': [],
        'memory_usage': [],
        'warmed_up': False
    }
    
    # Enhanced memory cleanup
    def optimized_cleanup_memory(self):
        """Enhanced memory cleanup with detailed monitoring"""
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_before = torch.cuda.memory_reserved() / (1024 ** 3)
            
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            allocated_after = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_after = torch.cuda.memory_reserved() / (1024 ** 3)
            
            freed_memory = (allocated_before - allocated_after) + (reserved_before - reserved_after)
            logger.debug(f"🧹 Memory cleanup: freed {freed_memory:.2f}GB")
        
        gc.collect()
    
    # Pipeline warmup
    def warmup_pipeline(self, quick_warmup=True):
        """Warm up pipeline for faster inference"""
        if self._performance_metrics.get('warmed_up', False):
            return
            
        logger.info("🔥 Starting pipeline warmup...")
        warmup_start = time.time()
        
        # Ensure models are loaded
        if not self.loaded:
            logger.info("📦 Loading checkpoints during warmup...")
            if hasattr(self, 'quantized') and self.quantized:
                self.load_quantized_checkpoint(self.checkpoint_dir)
            else:
                self.load_checkpoint(self.checkpoint_dir)
        
        # Quick warmup with small tensors
        try:
            with torch.no_grad():
                # Warm up text encoder
                dummy_texts = ["warmup text"]
                self.get_text_embeddings(dummy_texts)
                
                # Warm up lyric tokenizer
                dummy_lyrics = "[Verse]\nWarmup lyrics\n"
                self.tokenize_lyrics(dummy_lyrics)
                
                logger.info("✅ Pipeline warmup completed")
                
        except Exception as e:
            logger.warning(f"⚠️ Warmup encountered error (continuing): {e}")
        
        warmup_time = time.time() - warmup_start
        self._performance_metrics['warmup_time'] = warmup_time
        self._performance_metrics['warmed_up'] = True
        
        # Cleanup after warmup
        self.optimized_cleanup_memory()
        
        logger.info(f"✅ Pipeline warmup completed in {warmup_time:.2f}s")
    
    # Parallel preprocessing
    def parallel_preprocess(self, prompt: str, lyrics: str):
        """Process text and lyrics in parallel"""
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                text_future = executor.submit(lambda: self.get_text_embeddings([prompt]))
                lyric_future = executor.submit(lambda: self.tokenize_lyrics(lyrics))
                
                text_result = text_future.result()
                lyric_result = lyric_future.result()
                return text_result, lyric_result
        except Exception as e:
            logger.debug(f"Parallel preprocessing failed, using sequential: {e}")
            # Fallback to sequential
            text_result = self.get_text_embeddings([prompt])
            lyric_result = self.tokenize_lyrics(lyrics)
            return text_result, lyric_result
    
    # Get performance summary
    def get_performance_summary(self):
        """Get performance metrics summary"""
        metrics = self._performance_metrics.copy()
        
        if torch.cuda.is_available():
            metrics['current_gpu_allocated'] = torch.cuda.memory_allocated() / (1024 ** 3)
            metrics['current_gpu_reserved'] = torch.cuda.memory_reserved() / (1024 ** 3)
        
        return metrics
    
    # Bind methods to pipeline
    import types
    pipeline.optimized_cleanup_memory = types.MethodType(optimized_cleanup_memory, pipeline)
    pipeline.warmup_pipeline = types.MethodType(warmup_pipeline, pipeline)
    pipeline.parallel_preprocess = types.MethodType(parallel_preprocess, pipeline)
    pipeline.get_performance_summary = types.MethodType(get_performance_summary, pipeline)
    
    # Auto-warmup
    pipeline.warmup_pipeline(quick_warmup=True)

@app.post("/generate", response_model=ACEStepOutput)
async def generate_audio(input_data: ACEStepInput):
    """
    Generate audio using optimized ACEStep pipeline
    """
    global _performance_stats
    
    request_start = time.time()
    
    try:
        logger.info(f"🎵 Generating audio with optimizations enabled: {input_data.enable_optimizations}")
        logger.info(f"Prompt: {input_data.prompt[:50]}...")
        
        # Get or create cached optimized pipeline
        model_demo = get_or_create_pipeline(
            input_data.checkpoint_path,
            input_data.bf16,
            input_data.torch_compile,
            input_data.device_id,
            input_data.enable_optimizations
        )

        # Generate output path if not provided
        output_path = input_data.output_path or f"output_{uuid.uuid4().hex}.wav"

        # Prepare parameters for generation
        generation_params = {
            "audio_duration": input_data.audio_duration,
            "prompt": input_data.prompt,
            "lyrics": input_data.lyrics,
            "infer_step": input_data.infer_step,
            "guidance_scale": input_data.guidance_scale,
            "scheduler_type": input_data.scheduler_type,
            "cfg_type": input_data.cfg_type,
            "omega_scale": int(input_data.omega_scale),
            "manual_seeds": input_data.actual_seeds,
            "guidance_interval": input_data.guidance_interval,
            "guidance_interval_decay": input_data.guidance_interval_decay,
            "min_guidance_scale": input_data.min_guidance_scale,
            "use_erg_tag": input_data.use_erg_tag,
            "use_erg_lyric": input_data.use_erg_lyric,
            "use_erg_diffusion": input_data.use_erg_diffusion,
            "oss_steps": ", ".join(map(str, input_data.oss_steps)) if input_data.oss_steps else "",
            "guidance_scale_text": input_data.guidance_scale_text,
            "guidance_scale_lyric": input_data.guidance_scale_lyric,
            "save_path": output_path
        }

        # Use parallel preprocessing if enabled and optimizations are enabled
        if input_data.enable_optimizations and input_data.parallel_preprocessing and hasattr(model_demo, 'parallel_preprocess'):
            try:
                logger.debug("🔄 Using parallel preprocessing...")
                # Pre-process text and lyrics in parallel (this doesn't replace the generation)
                model_demo.parallel_preprocess(input_data.prompt, input_data.lyrics)
            except Exception as e:
                logger.debug(f"Parallel preprocessing failed: {e}")

        # Run pipeline generation
        result = model_demo(**generation_params)
        
        # Get performance metrics if available
        performance_metrics = None
        if hasattr(model_demo, 'get_performance_summary'):
            performance_metrics = model_demo.get_performance_summary()
        
        request_time = time.time() - request_start
        
        # Update global stats
        _performance_stats["requests_processed"] += 1
        _performance_stats["total_inference_time"] += request_time
        _performance_stats["avg_response_time"] = (
            _performance_stats["total_inference_time"] / _performance_stats["requests_processed"]
        )

        # Enhanced cleanup if available
        if hasattr(model_demo, 'optimized_cleanup_memory'):
            model_demo.optimized_cleanup_memory()

        logger.info(f"✅ Audio generation completed successfully in {request_time:.2f}s: {output_path}")
        
        return ACEStepOutput(
            status="success",
            output_path=output_path,
            message=f"Audio generated successfully in {request_time:.2f}s",
            performance_metrics=performance_metrics
        )

    except Exception as e:
        request_time = time.time() - request_start
        error_msg = f"Error generating audio after {request_time:.2f}s: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    """Enhanced health check with optimization status"""
    global _performance_stats
    
    health_info = {
        "status": "healthy",
        "cached_models": len(_model_cache),
        "cache_keys": list(_model_cache.keys()),
        "optimizations_enabled": True,
        "performance_stats": _performance_stats.copy()
    }
    
    # Add GPU memory info if available
    try:
        import torch
        if torch.cuda.is_available():
            health_info["gpu_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
                "device_count": torch.cuda.device_count()
            }
    except ImportError:
        pass
    
    return health_info

@app.post("/clear_cache")
async def clear_model_cache():
    """Clear the optimized model cache to free memory"""
    global _model_cache, _performance_stats
    
    cleared_count = len(_model_cache)
    
    # Properly cleanup optimized pipelines
    for cache_key, pipeline in _model_cache.items():
        try:
            # Call cleanup if available
            if hasattr(pipeline, 'optimized_cleanup_memory'):
                pipeline.optimized_cleanup_memory()
        except Exception as e:
            logger.warning(f"Error cleaning up pipeline {cache_key}: {e}")
    
    _model_cache.clear()
    
    # Reset performance stats
    _performance_stats = {
        "requests_processed": 0,
        "total_inference_time": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "avg_response_time": 0.0
    }
    
    logger.info(f"🧹 Cleared {cleared_count} cached optimized models and reset stats")
    
    return {
        "status": "success",
        "message": f"Cleared {cleared_count} cached optimized models and reset performance stats"
    }

@app.get("/performance", response_model=PerformanceStats)
async def get_performance_stats():
    """Get detailed performance statistics"""
    global _performance_stats
    
    return PerformanceStats(
        requests_processed=_performance_stats["requests_processed"],
        total_inference_time=_performance_stats["total_inference_time"],
        cache_hits=_performance_stats["cache_hits"],
        cache_misses=_performance_stats["cache_misses"],
        avg_response_time=_performance_stats["avg_response_time"],
        cached_models=len(_model_cache),
        cache_keys=list(_model_cache.keys())
    )

@app.post("/warmup")
async def warmup_pipeline(
    checkpoint_path: str,
    bf16: bool = True,
    torch_compile: bool = False,
    device_id: int = 0,
    enable_optimizations: bool = True
):
    """
    Explicitly warm up a pipeline configuration for faster subsequent requests
    """
    try:
        start_time = time.time()
        
        pipeline = get_or_create_pipeline(
            checkpoint_path, bf16, torch_compile, device_id, enable_optimizations
        )
        
        # Ensure it's warmed up
        if hasattr(pipeline, 'warmup_pipeline'):
            if not pipeline._performance_metrics.get('warmed_up', False):
                pipeline.warmup_pipeline(quick_warmup=False)  # Full warmup
        
        warmup_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": f"Pipeline warmed up successfully in {warmup_time:.2f}s",
            "warmup_time": warmup_time,
            "pipeline_ready": True
        }
        
    except Exception as e:
        logger.error(f"Failed to warmup pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")

@app.get("/optimization_info")
async def get_optimization_info():
    """Get information about available optimizations"""
    info = {
        "available_optimizations": [
            "Model preloading and warmup",
            "Enhanced memory management",
            "Parallel preprocessing",
            "Global model caching"
        ],
        "phase_1_implemented": True,
        "recommended_settings": {
            "bf16": True,
            "torch_compile": True,
            "enable_optimizations": True,
            "quick_warmup": True,
            "parallel_preprocessing": True
        },
        "performance_tips": [
            "Use warmup endpoint before first request for best performance",
            "Keep bf16=True for optimal memory usage",
            "Enable torch_compile for sustained workloads", 
            "Monitor /performance endpoint for optimization effectiveness"
        ]
    }
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
