class SpatialMixin:
    """
    Base class for task focusing on assessing spatial perception and understanding.
    This class provides methods to generate spatial relations between entities.
    """
    def generate_spatial_relation(self, physics):
        target_entity_name = self.target_entity
        target_entity_pos = self.entities[target_entity_name].get_xpos(physics)
    
    def build_from_config(self, config, eval=False):
        super().build_from_config(config, eval)
        self.generate_spatial_relation()
