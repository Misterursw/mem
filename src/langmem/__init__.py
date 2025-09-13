from langmem.knowledge import (
    create_manage_memory_tool,
    create_memory_manager,
    create_memory_searcher,
    create_memory_store_manager,
    create_search_memory_tool,
    create_thread_extractor,
)
from langmem.prompts.optimization import (
    Prompt,
    create_multi_prompt_optimizer,
    create_prompt_optimizer,
)
from langmem.reflection import ReflectionExecutor

# 医学模块 - 可选导入
try:
    from langmem.medical import (
        create_medical_memory_manager,
        MedicalImageMemory,
        BoneAgeMemory,
        DiagnosisCorrection,
    )
    _medical_available = True
except ImportError:
    _medical_available = False

__all__ = [
    "create_memory_manager",
    "create_memory_store_manager",
    "create_manage_memory_tool",
    "create_search_memory_tool",
    "create_thread_extractor",
    "create_multi_prompt_optimizer",
    "create_prompt_optimizer",
    "create_memory_searcher",
    "ReflectionExecutor",
    "Prompt",
]

# 添加医学模块到__all__（如果可用）
if _medical_available:
    __all__.extend([
        "create_medical_memory_manager",
        "MedicalImageMemory",
        "BoneAgeMemory", 
        "DiagnosisCorrection",
    ])
