from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from acestep.pipeline_ace_step import ACEStepPipeline
import uuid
from loguru import logger

app = FastAPI(title="ACEStep Pipeline API")

# Global model cache to avoid loading models on every request
_model_cache = {}

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

class ACEStepOutput(BaseModel):
    status: str
    output_path: Optional[str]
    message: str

def get_or_create_pipeline(checkpoint_path: str, bf16: bool, torch_compile: bool, device_id: int) -> ACEStepPipeline:
    """Get cached pipeline or create new one if not exists"""
    # Create cache key from parameters
    cache_key = f"{checkpoint_path}_{bf16}_{torch_compile}_{device_id}"
    
    if cache_key not in _model_cache:
        logger.info(f"Creating new pipeline for cache key: {cache_key}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            dtype="bfloat16" if bf16 else "float32",
            torch_compile=torch_compile,
        )
        
        _model_cache[cache_key] = pipeline
        logger.info(f"Pipeline cached with key: {cache_key}")
    else:
        logger.info(f"Using cached pipeline for key: {cache_key}")
    
    return _model_cache[cache_key]

@app.post("/generate", response_model=ACEStepOutput)
async def generate_audio(input_data: ACEStepInput):
    try:
        logger.info(f"Generating audio with prompt: {input_data.prompt[:50]}...")
        
        # Get or create cached pipeline
        model_demo = get_or_create_pipeline(
            input_data.checkpoint_path,
            input_data.bf16,
            input_data.torch_compile,
            input_data.device_id
        )

        # Generate output path if not provided
        output_path = input_data.output_path or f"output_{uuid.uuid4().hex}.wav"

        # Run pipeline with proper parameter names
        model_demo(
            audio_duration=input_data.audio_duration,
            prompt=input_data.prompt,
            lyrics=input_data.lyrics,
            infer_step=input_data.infer_step,
            guidance_scale=input_data.guidance_scale,
            scheduler_type=input_data.scheduler_type,
            cfg_type=input_data.cfg_type,
            omega_scale=int(input_data.omega_scale),
            manual_seeds=input_data.actual_seeds,
            guidance_interval=input_data.guidance_interval,
            guidance_interval_decay=input_data.guidance_interval_decay,
            min_guidance_scale=input_data.min_guidance_scale,
            use_erg_tag=input_data.use_erg_tag,
            use_erg_lyric=input_data.use_erg_lyric,
            use_erg_diffusion=input_data.use_erg_diffusion,
            oss_steps=", ".join(map(str, input_data.oss_steps)) if input_data.oss_steps else "",
            guidance_scale_text=input_data.guidance_scale_text,
            guidance_scale_lyric=input_data.guidance_scale_lyric,
            save_path=output_path
        )

        logger.info(f"Audio generation completed successfully: {output_path}")
        return ACEStepOutput(
            status="success",
            output_path=output_path,
            message="Audio generated successfully"
        )

    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cached_models": len(_model_cache),
        "cache_keys": list(_model_cache.keys())
    }

@app.post("/clear_cache")
async def clear_model_cache():
    """Clear the model cache to free memory"""
    global _model_cache
    cleared_count = len(_model_cache)
    _model_cache.clear()
    logger.info(f"Cleared {cleared_count} cached models")
    return {
        "status": "success",
        "message": f"Cleared {cleared_count} cached models"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
