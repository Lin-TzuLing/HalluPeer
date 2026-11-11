import datetime

from .taxonomy_generator import TaxonomyGenerator
from .review_aspect_tagger import ReviewAspectTagger
from .review_filter import ReviewFilter
from .template_constructor import TemplateConstructor
from .template_injector import TemplateInjector
from .post_checker import PostChecker


AGENTS = {
    "taxonomy_generator": TaxonomyGenerator,
    "review_aspect_tagger": ReviewAspectTagger,
    "review_filter": ReviewFilter,
    "templates_constructor": TemplateConstructor,
    "templates_injector": TemplateInjector,
    "post_checker": PostChecker,
}


def load_agent(agent: str, cfg: dict) -> object:
    """
    Load an agent based on the specified task.
    
    Args:
        agent (str): The task name for which to load the agent.
        cfg (dict): Configuration dictionary containing agent settings.
        
    Returns:
        object: An instance of the agent corresponding to the task.
    """
    
    if agent not in AGENTS:
        raise ValueError(f"Unsupported agent: {agent}. Supported agents are: {list(AGENTS.keys())}")
    
    agent_cfg = cfg.agents[agent]
    
    # Create a copy of agent_cfg without the explicitly passed parameters
    kwargs = {k: v for k, v in agent_cfg.items() 
              if k not in ['model_name', 'max_new_tokens', 'temperature', 'port']}
    
    print("="*50)
    print(f"Model Loading - {datetime.datetime.now()}")
    
    agent = AGENTS[agent](model_name=agent_cfg.model_name,
                       max_new_tokens=agent_cfg.max_new_tokens,
                       temperature=agent_cfg.temperature,
                       port=agent_cfg.port,
                       **kwargs
                       )
    
    agent.verbose_print()
    
    print(f"Model Loaded - {datetime.datetime.now()}")
    print("="*50)
    
    return agent