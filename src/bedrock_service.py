"""
AWS Bedrock Unified Image Service
================================
Supports multiple AI models with a unified interface.
Just change BEDROCK_*_MODEL env vars to switch models - no code changes!

IMPORTANT: Bedrock is in us-east-1 region (separate from S3 in ap-south-1)

Models supported:
- Amazon Nova Canvas (BG removal, inpainting, outpainting)
- Stability AI Image Services (upscaling: fast, conservative, creative)
- Stable Diffusion 3.5 Large (image-to-image transformation)
- Amazon Titan Image Generator V2/V1 (legacy)
"""
from __future__ import annotations
import os
import json
import base64
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from PIL import Image
import io

from .config import get_config

load_dotenv()
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    AMAZON_NOVA = "amazon-nova"
    AMAZON_TITAN = "amazon-titan"
    STABILITY_AI = "stability-ai"
    STABILITY_SERVICES = "stability-services"


class Operation(Enum):
    BACKGROUND_REMOVAL = "background_removal"
    UPSCALE_FAST = "upscale_fast"
    UPSCALE_CONSERVATIVE = "upscale_conservative"
    UPSCALE_CREATIVE = "upscale_creative"
    LIGHTING_FIX = "lighting_fix"
    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"
    IMAGE_VARIATION = "image_variation"
    TEXT_TO_IMAGE = "text_to_image"


@dataclass
class ModelConfig:
    model_id: str
    provider: ModelProvider
    supported_operations: List[Operation]
    cost_per_operation: Dict[str, float]
    max_input_size: int = 1024
    max_output_size: int = 2048


@dataclass
class BedrockCallResult:
    success: bool
    image: Optional[Image.Image] = None
    operation: str = ""
    model_id: str = ""
    latency_ms: int = 0
    estimated_cost: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "estimated_cost_usd": self.estimated_cost,
            "error": self.error,
        }


# =============================================================================
# MODEL DEFINITIONS - Updated with available models
# =============================================================================

# CRITICAL: Nova Pro/Lite/Micro are TEXT UNDERSTANDING models - they CANNOT generate/modify images!
# They can only ANALYZE images. For image generation, you need:
#   - amazon.nova-canvas-v1:0 (requires special access)
#   - amazon.titan-image-generator-v2:0 (requires special access)

IMAGE_GENERATION_MODELS = {
    "amazon.nova-canvas-v1:0",
    "amazon.titan-image-generator-v2:0",
    "amazon.titan-image-generator-v1",
    "stability.sd3-5-large-v1:0",
    "us.stability.sd3-5-large-v1:0",  # Inference profile
    "stability.stable-image-ultra-v1:0",
    "stability.stable-image-core-v1:0",
    "stability.stable-diffusion-xl-v1",
    "stability.stable-conservative-upscale-v1:0",
    "us.stability.stable-conservative-upscale-v1:0",  # Inference profile
    "us.stability.stable-image-conservative-upscale-v1:0", # Inference profile
    "us.stability.stable-image-remove-background-v1:0",
}

TEXT_UNDERSTANDING_MODELS = {
    "amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-2-lite-v1:0",
    "amazon.nova-micro-v1:0",
}

# Model ID mapping: friendly name -> actual inference profile ID
MODEL_ID_MAPPING = {
    # Stability upscale models require inference profiles
    "stability.stable-conservative-upscale-v1:0": "us.stability.stable-conservative-upscale-v1:0",
    "stability.stable-creative-upscale-v1:0": "us.stability.stable-creative-upscale-v1:0",
    "stability.stable-fast-upscale-v1:0": "us.stability.stable-fast-upscale-v1:0",
    # Stability image manipulation models
    "stability.stable-image-remove-background-v1:0": "us.stability.stable-image-remove-background-v1:0",
    "stability.stable-image-inpaint-v1:0": "us.stability.stable-image-inpaint-v1:0",
    "stability.stable-outpaint-v1:0": "us.stability.stable-outpaint-v1:0",
}

AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    # Amazon Nova Canvas (IMAGE GENERATION - requires special access)
    "amazon.nova-canvas-v1:0": ModelConfig(
        model_id="amazon.nova-canvas-v1:0",
        provider=ModelProvider.AMAZON_NOVA,
        supported_operations=[
            Operation.BACKGROUND_REMOVAL, Operation.INPAINTING, Operation.OUTPAINTING,
            Operation.TEXT_TO_IMAGE, Operation.IMAGE_VARIATION, Operation.LIGHTING_FIX,
        ],
        cost_per_operation={
            "background_removal": 0.04, "inpainting": 0.04, "outpainting": 0.04,
            "text_to_image": 0.04, "image_variation": 0.04, "lighting_fix": 0.04,
        },
        max_input_size=1408,
        max_output_size=2048,
    ),
    # Amazon Nova TEXT UNDERSTANDING Models (CANNOT generate images - only analyze)
    "amazon.nova-pro-v1:0": ModelConfig(
        model_id="amazon.nova-pro-v1:0",
        provider=ModelProvider.AMAZON_NOVA,
        supported_operations=[],  # No image generation operations
        cost_per_operation={},
        max_input_size=1408,
        max_output_size=2048,
    ),
    "amazon.nova-lite-v1:0": ModelConfig(
        model_id="amazon.nova-lite-v1:0",
        provider=ModelProvider.AMAZON_NOVA,
        supported_operations=[],  # No image generation operations
        cost_per_operation={},
        max_input_size=1024,
        max_output_size=1536,
    ),
    "amazon.nova-2-lite-v1:0": ModelConfig(
        model_id="amazon.nova-2-lite-v1:0",
        provider=ModelProvider.AMAZON_NOVA,
        supported_operations=[],  # No image generation operations
        cost_per_operation={},
        max_input_size=1024,
        max_output_size=1536,
    ),
    "amazon.nova-micro-v1:0": ModelConfig(
        model_id="amazon.nova-micro-v1:0",
        provider=ModelProvider.AMAZON_NOVA,
        supported_operations=[],  # No image generation operations
        cost_per_operation={},
        max_input_size=768,
        max_output_size=1024,
    ),
    # Titan Image Generator (IMAGE GENERATION - requires special access)
    "amazon.titan-image-generator-v2:0": ModelConfig(
        model_id="amazon.titan-image-generator-v2:0",
        provider=ModelProvider.AMAZON_TITAN,
        supported_operations=[
            Operation.BACKGROUND_REMOVAL, Operation.INPAINTING, Operation.OUTPAINTING,
            Operation.IMAGE_VARIATION, Operation.TEXT_TO_IMAGE,
        ],
        cost_per_operation={
            "background_removal": 0.01, "inpainting": 0.012, "outpainting": 0.012,
            "image_variation": 0.01, "text_to_image": 0.008,
        },
        max_input_size=1408,
    ),
    # Stability AI Models (Available in us-east-1)
    "stability.sd3-5-large-v1:0": ModelConfig(
        model_id="us.stability.sd3-5-large-v1:0",  # Use inference profile
        provider=ModelProvider.STABILITY_AI,
        supported_operations=[
            Operation.TEXT_TO_IMAGE, Operation.IMAGE_VARIATION, Operation.LIGHTING_FIX,
        ],
        cost_per_operation={
            "text_to_image": 0.065, "image_variation": 0.065, "lighting_fix": 0.065,
        },
        max_input_size=1024,
        max_output_size=2048,
    ),
    "stability.stable-conservative-upscale-v1:0": ModelConfig(
        model_id="us.stability.stable-conservative-upscale-v1:0",  # Use inference profile
        provider=ModelProvider.STABILITY_SERVICES,
        supported_operations=[
            Operation.UPSCALE_CONSERVATIVE,
        ],
        cost_per_operation={
            "upscale_conservative": 0.04,
        },
        max_input_size=1024,
        max_output_size=4096,
    ),
    "stability.stable-creative-upscale-v1:0": ModelConfig(
        model_id="us.stability.stable-creative-upscale-v1:0",
        provider=ModelProvider.STABILITY_SERVICES,
        supported_operations=[Operation.UPSCALE_CREATIVE],
        cost_per_operation={"upscale_creative": 0.04},
        max_input_size=1024,
        max_output_size=4096,
    ),
    "stability.stable-fast-upscale-v1:0": ModelConfig(
        model_id="us.stability.stable-fast-upscale-v1:0",
        provider=ModelProvider.STABILITY_SERVICES,
        supported_operations=[Operation.UPSCALE_FAST],
        cost_per_operation={"upscale_fast": 0.02},
        max_input_size=1024,
        max_output_size=4096,
    ),
    # Stability Background Removal
    "stability.stable-image-remove-background-v1:0": ModelConfig(
        model_id="us.stability.stable-image-remove-background-v1:0",
        provider=ModelProvider.STABILITY_SERVICES,
        supported_operations=[Operation.BACKGROUND_REMOVAL],
        cost_per_operation={"background_removal": 0.04},
        max_input_size=1024,
        max_output_size=4096,
    ),
}

RECOMMENDED_MODELS = {
    Operation.BACKGROUND_REMOVAL: "stability.stable-image-remove-background-v1:0",
    Operation.UPSCALE_FAST: "stability.stable-fast-upscale-v1:0",
    Operation.UPSCALE_CONSERVATIVE: "stability.stable-conservative-upscale-v1:0",
    Operation.UPSCALE_CREATIVE: "stability.stable-creative-upscale-v1:0",
    Operation.LIGHTING_FIX: "amazon.nova-canvas-v1:0",
    Operation.INPAINTING: "amazon.nova-canvas-v1:0",
    Operation.OUTPAINTING: "amazon.nova-canvas-v1:0",
    Operation.IMAGE_VARIATION: "amazon.nova-canvas-v1:0",
    Operation.TEXT_TO_IMAGE: "amazon.nova-canvas-v1:0",
}


# =============================================================================
# REQUEST FORMATTERS
# =============================================================================
class RequestFormatter(ABC):
    @abstractmethod
    def format_request(self, operation: Operation, params: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def parse_response(self, response: Dict[str, Any]) -> Optional[Image.Image]:
        pass

ECOMMERCE_NEGATIVE = (
    "altered product details, changed text, distorted logo, morphed shape, "
    "different color, new elements, artistic interpretation, illustration, "
    "painting, cartoon, cgi, smoothing, blurred texture, jpeg artifacts, "
    "noise, grain, low resolution"
)

class NovaCanvasFormatter(RequestFormatter):
    def format_request(self, operation: Operation, params: Dict[str, Any]) -> Dict[str, Any]:
        base_cfg = {
            "numberOfImages": 1, 
            "quality": params.get("quality", "standard"), 
            "seed": params.get("seed", 42)
        }
        
        if operation == Operation.BACKGROUND_REMOVAL:
            return {"taskType": "BACKGROUND_REMOVAL", "backgroundRemovalParams": {"images": params["image_base64"]}}
        
        if operation == Operation.INPAINTING:
            # OPTIMIZED: Focus on texture and blending, strictly forbidding new objects
            prompt = params.get("prompt", "true color reproduction, neutral white balance, color consistency across product, enhance the quality")
            req = {
                "taskType": "INPAINTING", 
                "inPaintingParams": {
                    "images": params["image_base64"], 
                    "text": prompt,
                    "negativeText": ECOMMERCE_NEGATIVE
                }, 
                "imageGenerationConfig": base_cfg
            }
            if "mask_base64" in params:
                req["inPaintingParams"]["maskImage"] = params["mask_base64"]
            elif "mask_prompt" in params:
                req["inPaintingParams"]["maskPrompt"] = params["mask_prompt"]
            return req
        
        if operation == Operation.OUTPAINTING:
            # OPTIMIZED: Ensure the extension matches the studio setting, not the world
            prompt = params.get("prompt", "true color reproduction, neutral white balance, color consistency across product, enhance the quality")
            return {
                "taskType": "OUTPAINTING", 
                "outPaintingParams": {
                    "images": params["image_base64"], 
                    "text": prompt, 
                    "maskImage": params.get("mask_base64"), 
                    "outPaintingMode": params.get("mode", "DEFAULT"),
                    "negativeText": "cluttered, busy, random objects, people, animals, bright colors"
                }, 
                "imageGenerationConfig": base_cfg
            }
        
        if operation == Operation.TEXT_TO_IMAGE:
            # Unlikely to be used for enhancement, but kept for completeness
            return {"taskType": "TEXT_IMAGE", "textToImageParams": {"text": params.get("prompt", "professional product photo"), "negativeText": params.get("negative_prompt", ECOMMERCE_NEGATIVE)}, "imageGenerationConfig": {**base_cfg, "width": params.get("width", 1024), "height": params.get("height", 1024), "cfgScale": params.get("cfg_scale", 7.0)}}
        
        if operation in [Operation.IMAGE_VARIATION, Operation.LIGHTING_FIX]:
            # CRITICAL OPTIMIZATION:
            if operation == Operation.LIGHTING_FIX:
                # Prompt focuses purely on "photographic properties", not "content"
                prompt = params.get("prompt", "true color reproduction, neutral white balance, color consistency across product, enhance the quality")
                similarity = params.get("similarity", 0.99) # Increased to 0.99 for lighting only
            else:
                prompt = params.get("prompt", "true color reproduction, neutral white balance, color consistency across product, enhance the quality")
                similarity = params.get("similarity", 0.96) 
            
            return {
                "taskType": "IMAGE_VARIATION",
                "imageVariationParams": {
                    "images": [params["image_base64"]],
                    "text": prompt,
                    "negativeText": ECOMMERCE_NEGATIVE,
                    "similarityStrength": similarity
                },
                "imageGenerationConfig": base_cfg
            }
        
        raise ValueError(f"Unsupported: {operation}")
    
    def parse_response(self, response: Dict[str, Any]) -> Optional[Image.Image]:
        if "images" in response and response["images"]:
            return Image.open(io.BytesIO(base64.b64decode(response["images"][0])))
        return None


class StabilityServicesFormatter(RequestFormatter):
    def format_request(self, operation: Operation, params: Dict[str, Any]) -> Dict[str, Any]:
        if operation == Operation.UPSCALE_FAST:
            return {"image": params["image_base64"], "output_format": "png"}
        
        if operation == Operation.UPSCALE_CONSERVATIVE:
            # OPTIMIZED: Lowered creativity to 0.2 and focused prompt on "fidelity"
            return {
                "image": params["image_base64"], 
                "prompt": params.get("prompt", "true color reproduction, neutral white balance, color consistency across product, enhance the quality"), 
                "negative_prompt": ECOMMERCE_NEGATIVE,
                "creativity": params.get("creativity", 0.20), # Keep this low to prevent hallucination
                "output_format": "png", 
                "seed": params.get("seed", 0)
            }
            
        if operation == Operation.UPSCALE_CREATIVE:
            # Even for "creative", we want to constrain it to the product
            return {
                "image": params["image_base64"], 
                "prompt": params.get("prompt", "true color reproduction, neutral white balance, color consistency across product, enhance the quality"), 
                "creativity": params.get("creativity", 0.25), # Lowered from 0.3
                "negative_prompt": params.get("negative_prompt", ECOMMERCE_NEGATIVE), 
                "output_format": "png", 
                "seed": params.get("seed", 0)
            }
            
        if operation == Operation.BACKGROUND_REMOVAL:
            return {"image": params["image_base64"], "output_format": "png"}
        raise ValueError(f"Unsupported: {operation}")
    
    def parse_response(self, response: Dict[str, Any]) -> Optional[Image.Image]:
        if "images" in response and response["images"]:
            return Image.open(io.BytesIO(base64.b64decode(response["images"][0])))
        if "image" in response:
            return Image.open(io.BytesIO(base64.b64decode(response["image"])))
        return None


class StableDiffusionFormatter(RequestFormatter):
    def format_request(self, operation: Operation, params: Dict[str, Any]) -> Dict[str, Any]:
        if operation == Operation.TEXT_TO_IMAGE:
            return {"prompt": params.get("prompt", "professional product"), "negative_prompt": params.get("negative_prompt", ECOMMERCE_NEGATIVE), "seed": params.get("seed", 0), "output_format": "png"}
        
        if operation in [Operation.IMAGE_VARIATION, Operation.LIGHTING_FIX]:
            # OPTIMIZED: Using very precise technical terms
            if operation == Operation.LIGHTING_FIX:
                prompt = params.get("prompt", "true color reproduction, neutral white balance, color consistency across product, enhance the quality")
                # Strength represents "how much to change". 0.15 is safe for lighting.
                strength = params.get("strength", 0.15) 
            else:
                prompt = params.get("prompt", "true color reproduction, neutral white balance, color consistency across product, enhance the quality")
                strength = params.get("strength", 0.25) 
            
            return {
                "prompt": prompt,
                "image": params["image_base64"],
                "strength": strength,
                "negative_prompt": params.get("negative_prompt", ECOMMERCE_NEGATIVE),
                "seed": params.get("seed", 0),
                "output_format": "png"
            }
        raise ValueError(f"Unsupported: {operation}")
    
    def parse_response(self, response: Dict[str, Any]) -> Optional[Image.Image]:
        if "images" in response and response["images"]:
            return Image.open(io.BytesIO(base64.b64decode(response["images"][0])))
        return None


class TitanImageFormatter(RequestFormatter):
    def format_request(self, operation: Operation, params: Dict[str, Any]) -> Dict[str, Any]:
        base_cfg = {"numberOfImages": 1, "quality": params.get("quality", "standard"), "cfgScale": params.get("cfg_scale", 8.0), "seed": params.get("seed", 42)}
        if params.get("width"):
            base_cfg["width"] = params["width"]
        if params.get("height"):
            base_cfg["height"] = params["height"]
        
        if operation == Operation.BACKGROUND_REMOVAL:
            return {"taskType": "BACKGROUND_REMOVAL", "backgroundRemovalParams": {"images": params["image_base64"]}}
            
        if operation == Operation.TEXT_TO_IMAGE:
            return {"taskType": "TEXT_IMAGE", "textToImageParams": {"text": params.get("prompt", "product"), "negativeText": params.get("negative_prompt", ECOMMERCE_NEGATIVE)}, "imageGenerationConfig": base_cfg}
            
        if operation in [Operation.IMAGE_VARIATION, Operation.LIGHTING_FIX]:
            if operation == Operation.LIGHTING_FIX:
                prompt = params.get("prompt", "true color reproduction, neutral white balance, color consistency across product, enhance the quality")
                similarity = params.get("similarity", 0.99) # Extremely high to prevent shape changes
            else:
                prompt = params.get("prompt", "true color reproduction, neutral white balance, color consistency across product, enhance the quality")
                similarity = params.get("similarity", 0.96)
            
            return {
                "taskType": "IMAGE_VARIATION",
                "imageVariationParams": {
                    "text": prompt,
                    "negativeText": params.get("negative_prompt", ECOMMERCE_NEGATIVE),
                    "images": [params["image_base64"]],
                    "similarityStrength": similarity
                },
                "imageGenerationConfig": base_cfg
            }
            
        if operation == Operation.INPAINTING:
            prompt = params.get("prompt", "restore texture, seamless repair, high quality match")
            return {
                "taskType": "INPAINTING",
                "inPaintingParams": {
                    "text": prompt,
                    "negativeText": params.get("negative_prompt", ECOMMERCE_NEGATIVE),
                    "images": params["image_base64"],
                    "maskImage": params.get("mask_base64"),
                    "maskPrompt": params.get("mask_prompt")
                },
                "imageGenerationConfig": base_cfg
            }
        raise ValueError(f"Unsupported: {operation}")
    
    def parse_response(self, response: Dict[str, Any]) -> Optional[Image.Image]:
        if "images" in response and response["images"]:
            return Image.open(io.BytesIO(base64.b64decode(response["images"][0])))
        return None


FORMATTERS: Dict[ModelProvider, RequestFormatter] = {
    ModelProvider.AMAZON_NOVA: NovaCanvasFormatter(),
    ModelProvider.AMAZON_TITAN: TitanImageFormatter(),
    ModelProvider.STABILITY_AI: StableDiffusionFormatter(),
    ModelProvider.STABILITY_SERVICES: StabilityServicesFormatter(),
}


# =============================================================================
# MAIN SERVICE CLASS
# =============================================================================
class BedrockService:
    """
    Unified Bedrock Image Service
    
    IMPORTANT: Bedrock is in us-east-1 region (separate from S3 in ap-south-1)
    
    Just set these env vars to switch models:
      BEDROCK_BG_MODEL=amazon.nova-canvas-v1:0
      BEDROCK_UPSCALE_MODEL=stability.si.conservative-upscale-v2:0
      BEDROCK_LIGHTING_MODEL=amazon.nova-canvas-v1:0
    """
    
    def __init__(self, default_region: str = None):
        self.config = get_config()
        # IMPORTANT: Bedrock is always in us-east-1
        self.region = self.config.hybrid.bedrock_region or "us-east-1"
        self._client = None
        self._daily_cost = 0.0
        self._call_count = 0
        self._last_reset = datetime.now().date()
        self._call_history: List[Dict] = []
        
        # Load model preferences from env (using available models in us-east-1)
        self.default_bg_model = os.getenv("BEDROCK_BG_MODEL", "stability.stable-image-remove-background-v1:0")
        self.default_upscale_model = os.getenv("BEDROCK_UPSCALE_MODEL", "stability.stable-conservative-upscale-v1:0")
        self.default_lighting_model = os.getenv("BEDROCK_LIGHTING_MODEL", "amazon.nova-canvas-v1:0")
        self.default_variation_model = os.getenv("BEDROCK_VARIATION_MODEL", "amazon.nova-canvas-v1:0")
        
        logger.info("=" * 60)
        logger.info("🤖 BedrockService Initialized (Multi-Model)")
        logger.info(f"   Region: {self.region}")
        logger.info(f"   Enabled: {self.config.hybrid.enable_bedrock}")
        logger.info(f"   BG Model: {self.default_bg_model}")
        logger.info(f"   Upscale Model: {self.default_upscale_model}")
        logger.info(f"   Lighting Model: {self.default_lighting_model}")
        logger.info(f"   Max Daily Cost: ${self.config.hybrid.max_daily_cost:.2f}")
        logger.info("=" * 60)
    
    @property
    def client(self):
        if self._client is None:
            try:
                ak = os.getenv("AWS_ACCESS_KEY_ID")
                sk = os.getenv("AWS_SECRET_ACCESS_KEY")
                
                if ak and sk:
                    self._client = boto3.client(
                        'bedrock-runtime', 
                        region_name=self.region, 
                        aws_access_key_id=ak, 
                        aws_secret_access_key=sk
                    )
                    logger.info(f"✅ Bedrock client initialized with credentials (region: {self.region})")
                else:
                    self._client = boto3.client('bedrock-runtime', region_name=self.region)
                    logger.info(f"✅ Bedrock client initialized with default credentials (region: {self.region})")
            except NoCredentialsError:
                logger.error("❌ AWS credentials not found for Bedrock")
            except Exception as e:
                logger.error(f"❌ Bedrock init failed: {e}")
        return self._client
    
    def _reset_daily_cost_if_needed(self):
        today = datetime.now().date()
        if today > self._last_reset:
            logger.info(f"📊 Daily reset. Prev: ${self._daily_cost:.4f} ({self._call_count} calls)")
            self._daily_cost, self._call_count, self._last_reset, self._call_history = 0.0, 0, today, []
    
    def _check_cost_limit(self) -> bool:
        self._reset_daily_cost_if_needed()
        max_cost = self.config.hybrid.max_daily_cost
        if self._daily_cost >= max_cost:
            logger.warning(f"⚠️ Daily limit: ${self._daily_cost:.4f} >= ${max_cost:.2f}")
            return False
        return True
    
    def _track_cost(self, model_id: str, operation: str, cost: float):
        self._daily_cost += cost
        self._call_count += 1
        self._call_history.append({"ts": datetime.now().isoformat(), "model": model_id, "op": operation, "cost": cost})
        logger.info(f"💰 ${cost:.4f} | Total: ${self._daily_cost:.4f} ({self._call_count})")
    
    def _image_to_base64(self, image: Image.Image, max_size: int = 1024) -> str:
        if max(image.size) > max_size:
            r = max_size / max(image.size)
            image = image.resize((int(image.size[0] * r), int(image.size[1] * r)), Image.LANCZOS)
        buf = io.BytesIO()
        if image.mode == 'RGBA':
            image.save(buf, format="PNG")
        else:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def is_available(self) -> bool:
        return self.config.hybrid.enable_bedrock and self.client is not None
    
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model config - checks original ID, mapped ID, and reverse lookup"""
        # First check if it's directly in AVAILABLE_MODELS
        if model_id in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_id]
        
        # Check if this is an inference profile ID - do reverse lookup
        for friendly_name, inference_profile in MODEL_ID_MAPPING.items():
            if model_id == inference_profile:
                return AVAILABLE_MODELS.get(friendly_name)
        
        # Then check if we need to map it forward
        actual_model_id = MODEL_ID_MAPPING.get(model_id, model_id)
        return AVAILABLE_MODELS.get(actual_model_id)
    
    def list_available_models(self) -> List[str]:
        return list(AVAILABLE_MODELS.keys())
    
    def get_recommended_model(self, operation: Operation) -> str:
        return RECOMMENDED_MODELS.get(operation, "amazon.nova-canvas-v1:0")
    
    def invoke(self, operation: Operation, image: Optional[Image.Image] = None, model_id: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> BedrockCallResult:
        """Universal invoke - auto-selects model if not specified"""
        start = time.time()
        params = params or {}
        
        if not model_id:
            model_id = self.get_recommended_model(operation)
            logger.info(f"🤖 Auto-selected model: {model_id} for {operation.value}")
        
        # Map friendly name to actual model ID
        actual_model_id = model_id
        # Check forward mapping (friendly → inference profile)
        if model_id in MODEL_ID_MAPPING:
            actual_model_id = MODEL_ID_MAPPING[model_id]
            logger.info(f"📝 Mapped {model_id} → {actual_model_id}")
        else:
            # Check if already an inference profile - use as-is
            for friendly, profile in MODEL_ID_MAPPING.items():
                if model_id == profile:
                    logger.info(f"📝 Using inference profile directly: {model_id}")
                    break
        
        # CRITICAL VALIDATION: Check if model can generate images
        if actual_model_id in TEXT_UNDERSTANDING_MODELS:
            error_msg = (
                f"❌ INVALID MODEL: {actual_model_id} is a TEXT UNDERSTANDING model that CANNOT generate/modify images.\n"
                f"   It can only ANALYZE images. For image generation/manipulation, you need:\n"
                f"   - amazon.nova-canvas-v1:0 (recommended)\n"
                f"   - amazon.titan-image-generator-v2:0\n"
                f"   Request access in AWS Bedrock Console: https://console.aws.amazon.com/bedrock/\n"
                f"   Falling back to LOCAL processing."
            )
            logger.error(error_msg)
            return BedrockCallResult(success=False, operation=operation.value, model_id=actual_model_id, error="Model cannot generate images")
        
        cfg = self.get_model_config(model_id)
        if not cfg:
            logger.error(f"❌ Unknown model: {model_id}")
            return BedrockCallResult(success=False, operation=operation.value, error=f"Unknown model: {model_id}")
        if operation not in cfg.supported_operations:
            logger.error(f"❌ {operation.value} not supported by {model_id}")
            return BedrockCallResult(success=False, operation=operation.value, model_id=actual_model_id, error=f"{operation.value} not supported by {model_id}")
        
        logger.info("=" * 60)
        logger.info(f"🚀 BEDROCK MODEL CALL")
        logger.info(f"   Operation: {operation.value}")
        logger.info(f"   Model: {actual_model_id}")
        logger.info(f"   Provider: {cfg.provider.value}")
        logger.info(f"   Model Type: {'IMAGE GENERATION' if actual_model_id in IMAGE_GENERATION_MODELS else 'TEXT UNDERSTANDING'}")
        logger.info(f"   Estimated Cost: ${cfg.cost_per_operation.get(operation.value, 0.02):.4f}")
        if image:
            logger.info(f"   Input Image: {image.size[0]}x{image.size[1]} ({image.mode})")
        logger.info("-" * 60)
        
        if not self.is_available():
            logger.warning("⚠️ Bedrock unavailable - check AWS credentials")
            return BedrockCallResult(success=False, operation=operation.value, model_id=model_id, error="Bedrock unavailable")
        if not self._check_cost_limit():
            logger.warning(f"⚠️ Daily cost limit reached: ${self._daily_cost:.2f}")
            return BedrockCallResult(success=False, operation=operation.value, model_id=model_id, error="Cost limit")
        
        try:
            if image:
                params["image_base64"] = self._image_to_base64(image, cfg.max_input_size)
                params["width"] = params.get("width", image.width)
                params["height"] = params.get("height", image.height)
            
            fmt = FORMATTERS[cfg.provider]
            req = fmt.format_request(operation, params)
            
            logger.info(f"📤 Sending request to Bedrock...")
            resp = self.client.invoke_model(modelId=actual_model_id, body=json.dumps(req), accept="application/json", contentType="application/json")
            body = json.loads(resp.get("body").read())
            logger.info(f"📦 Bedrock Response Keys: {list(body.keys())}")
            if "status" in body:
                 logger.info(f"📦 Status: {body.get('status')}")
            
            img = fmt.parse_response(body)
            
            if img:
                lat = int((time.time() - start) * 1000)
                cost = cfg.cost_per_operation.get(operation.value, 0.02)
                self._track_cost(actual_model_id, operation.value, cost)
                logger.info(f"✅ SUCCESS")
                logger.info(f"   Output Image: {img.size[0]}x{img.size[1]} ({img.mode})")
                logger.info(f"   Latency: {lat}ms")
                logger.info(f"   Cost: ${cost:.4f}")
                logger.info(f"💰 Running Total: ${self._daily_cost:.4f} ({self._call_count} calls today)")
                logger.info("=" * 60)
                return BedrockCallResult(success=True, image=img, operation=operation.value, model_id=actual_model_id, latency_ms=lat, estimated_cost=cost)
            raise ValueError("No image in response")
        except ClientError as e:
            err = e.response.get("Error", {})
            logger.error(f"❌ Bedrock API Error: {err.get('Code')}: {err.get('Message')}")
            logger.error("=" * 60)
            return BedrockCallResult(success=False, operation=operation.value, model_id=actual_model_id, latency_ms=int((time.time()-start)*1000), error=str(e))
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            logger.error("=" * 60)
            return BedrockCallResult(success=False, operation=operation.value, model_id=actual_model_id, latency_ms=int((time.time()-start)*1000), error=str(e))
    
    # Convenience methods
    def remove_background(self, image: Image.Image, model_id: str = None, **kw) -> BedrockCallResult:
        return self.invoke(Operation.BACKGROUND_REMOVAL, image, model_id or self.default_bg_model, kw)
    
    def upscale_image(self, image: Image.Image, mode: str = "conservative", model_id: str = None, **kw) -> BedrockCallResult:
        """Upscale image - uses dedicated upscale model if available, otherwise image variation"""
        model_id = model_id or self.default_upscale_model
        
        # Check if model supports dedicated upscaling
        if "upscale" in model_id.lower():
            return self.invoke(Operation.UPSCALE_CONSERVATIVE, image, model_id, kw)
        
        # Fallback to image variation with upscaling prompt
        kw["prompt"] = kw.get("prompt", "ultra high resolution, sharp details, 4K quality")
        kw["similarity"] = kw.get("similarity", 0.85)
        return self.invoke(Operation.IMAGE_VARIATION, image, model_id, kw)
    
    def fix_lighting(self, image: Image.Image, model_id: str = None, prompt: str = None, **kw) -> BedrockCallResult:
        kw["prompt"] = prompt or "professional studio lighting, perfect exposure"
        kw["similarity"] = kw.get("similarity", 0.6)
        return self.invoke(Operation.LIGHTING_FIX, image, model_id or self.default_lighting_model, kw)
    
    def create_variation(self, image: Image.Image, prompt: str = "high quality", model_id: str = None, similarity: float = 0.7, **kw) -> BedrockCallResult:
        kw["prompt"], kw["similarity"] = prompt, similarity
        return self.invoke(Operation.IMAGE_VARIATION, image, model_id or self.default_variation_model, kw)
    
    def inpaint(self, image: Image.Image, prompt: str, mask_image: Image.Image = None, mask_prompt: str = None, model_id: str = None, **kw) -> BedrockCallResult:
        kw["prompt"] = prompt
        if mask_image:
            kw["mask_base64"] = self._image_to_base64(mask_image)
        elif mask_prompt:
            kw["mask_prompt"] = mask_prompt
        return self.invoke(Operation.INPAINTING, image, model_id or "amazon.nova-canvas-v1:0", kw)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        self._reset_daily_cost_if_needed()
        return {"daily_cost_usd": round(self._daily_cost, 4), "calls": self._call_count, "max_cost": self.config.hybrid.max_daily_cost, "remaining": round(self.config.hybrid.max_daily_cost - self._daily_cost, 4), "available": self.is_available()}
    
    def get_cost_by_model(self) -> Dict[str, float]:
        c = {}
        for h in self._call_history:
            c[h["model"]] = c.get(h["model"], 0) + h["cost"]
        return {k: round(v, 4) for k, v in c.items()}
    
    def get_cost_by_operation(self) -> Dict[str, float]:
        c = {}
        for h in self._call_history:
            c[h["op"]] = c.get(h["op"], 0) + h["cost"]
        return {k: round(v, 4) for k, v in c.items()}


# =============================================================================
# FACTORY & HELPERS
# =============================================================================
def create_bedrock_service(region: str = None) -> BedrockService:
    return BedrockService(default_region=region)


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    cfg = AVAILABLE_MODELS.get(model_id)
    if not cfg:
        return None
    return {
        "model_id": cfg.model_id,
        "provider": cfg.provider.value,
        "operations": [op.value for op in cfg.supported_operations],
        "costs": cfg.cost_per_operation,
        "max_input": cfg.max_input_size,
    }


def list_models_by_provider(provider: str) -> List[str]:
    try:
        p = ModelProvider(provider)
    except ValueError:
        return []
    return [m for m, c in AVAILABLE_MODELS.items() if c.provider == p]


def get_cheapest_model_for_operation(operation: Operation) -> Optional[str]:
    cands = [(m, c.cost_per_operation.get(operation.value, float('inf'))) for m, c in AVAILABLE_MODELS.items() if operation in c.supported_operations]
    return min(cands, key=lambda x: x[1])[0] if cands else None