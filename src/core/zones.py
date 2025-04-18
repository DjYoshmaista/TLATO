# src/core/zones.py
"""
Neural Zone Module
Implements neural zones and groups with activation dynamics based on configuration.
"""

import torch
import random
import logging
# Import the specific config class and the logger setup
from src.utils.config import ZoneConfig
from src.utils.logger import setup_logger # Or directly use SemanticLogger if it's a class

# Get a logger for this module
logger = logging.getLogger(__name__) # Use standard logging; setup_logger configures the root

# Note: The original ZoneConfig dataclass is removed as it's now imported from src.utils.config

class NeuralZone:
    """
    Represents an individual neural zone with activation level and connections.

    Attributes:
        identifier (str): A unique identifier for the zone.
        config (ZoneConfig): Configuration settings for the zone.
        activation (torch.Tensor): The current activation level of the zone (scalar tensor).
        connections (list): A list of dictionaries, each representing a connection
                           to another NeuralZone.
    """

    def __init__(self, identifier: str, config: ZoneConfig = ZoneConfig()):
        """
        Initializes a NeuralZone instance.

        Args:
            identifier (str): The unique identifier for this zone.
            config (ZoneConfig, optional): Configuration object. Defaults to ZoneConfig().
        """
        self.identifier = identifier
        self.config = config
        # Ensure activation is a scalar tensor
        self.activation = torch.tensor([config.BASE_ACTIVATION], dtype=torch.float32)
        self.connections = [] # List to store outgoing connections

        logger.info(f"Initialized zone '{self.identifier}'", extra={
            'base_activation': self.config.BASE_ACTIVATION,
            'max_connections': self.config.MAX_CONNECTIONS
        })

    def link_zone(self, target: 'NeuralZone', strength: float = None):
        """
        Establishes a directed connection from this zone to a target zone.

        Args:
            target (NeuralZone): The target zone to connect to.
            strength (float, optional): The strength of the connection.
                                        Defaults to config.LINK_STRENGTH.

        Returns:
            bool: True if the connection was successfully added, False otherwise
                  (e.g., if the connection limit is reached).
        """
        if len(self.connections) >= self.config.MAX_CONNECTIONS:
            logger.warning(f"Connection limit ({self.config.MAX_CONNECTIONS}) reached for zone '{self.identifier}'. Cannot link to '{target.identifier}'.")
            return False

        # Use provided strength or default from config
        link_strength = strength if strength is not None else self.config.LINK_STRENGTH
        if not isinstance(link_strength, torch.Tensor):
             link_strength = torch.tensor(link_strength, dtype=torch.float32)

        connection_info = {
            'target': target,
            'strength': link_strength,
            'last_activated': None # Placeholder for potential future use (e.g., Hebbian learning)
        }
        self.connections.append(connection_info)
        logger.debug(f"Linked '{self.identifier}' -> '{target.identifier}' with strength {link_strength.item():.2f}")
        return True

    def update_activation(self, stimulus: torch.Tensor):
        """
        Updates the zone's activation based on external stimulus and connected zones.

        The activation is calculated as:
        activation = sigmoid(stimulus + current_activation + sum(conn_strength * target_activation))
        The result is clamped between 0 and 1.

        Args:
            stimulus (torch.Tensor): An external stimulus tensor (should be scalar or broadcastable).

        Returns:
            torch.Tensor: The updated activation tensor of the zone.
        """
        try:
            # Ensure stimulus is a tensor
            if not isinstance(stimulus, torch.Tensor):
                stimulus = torch.tensor(stimulus, dtype=torch.float32)

            # Aggregate influence from connected zones
            incoming_activation = torch.tensor(0.0, dtype=torch.float32)
            if self.connections:
                for conn in self.connections:
                    # Ensure target activation is accessible and is a tensor
                    target_activation = conn['target'].activation
                    if not isinstance(target_activation, torch.Tensor):
                         # Attempt to convert if possible, or log error
                         try:
                            target_activation = torch.tensor(target_activation, dtype=torch.float32)
                         except Exception:
                            logger.error(f"Invalid activation type in target zone '{conn['target'].identifier}'. Skipping connection.", exc_info=True)
                            continue

                    incoming_activation += conn['strength'] * target_activation

            # Calculate new activation using sigmoid and clamp
            # Combine stimulus, current activation, and incoming signals
            total_input = stimulus + self.activation + incoming_activation
            new_activation = torch.sigmoid(total_input)

            # Clamp the activation to be within [0, 1]
            self.activation = torch.clamp(new_activation, 0.0, 1.0)

            logger.debug(f"Zone '{self.identifier}' updated activation: {self.activation.item():.4f} (Stimulus: {stimulus.item():.4f}, Incoming: {incoming_activation.item():.4f})")
            return self.activation

        except Exception as e:
            logger.error(f"Activation update failed for zone '{self.identifier}': {str(e)}", exc_info=True)
            # Potentially re-raise or return current activation to prevent crash
            # raise # Uncomment to propagate the error
            return self.activation # Return current state on failure


class NeuralZoneGroup:
    """
    Manages a collection of interacting NeuralZone instances.

    Attributes:
        group_id (str): Identifier for the group.
        zones (dict): Dictionary mapping zone identifiers (str) to NeuralZone objects.
        config (ZoneConfig): Configuration applied to zones created within this group.
    """

    def __init__(self, group_id: str, capacity: int = 10, config: ZoneConfig = ZoneConfig()):
        """
        Initializes a NeuralZoneGroup.

        Args:
            group_id (str): The identifier for this group of zones.
            capacity (int, optional): The initial number of zones to create. Defaults to 10.
            config (ZoneConfig, optional): Configuration for zones in this group. Defaults to ZoneConfig().
        """
        self.group_id = group_id
        self.zones = {} # Dictionary {zone_id: NeuralZone}
        self.config = config

        if capacity <= 0:
             logger.warning(f"Attempted to create NeuralZoneGroup '{group_id}' with non-positive capacity ({capacity}). Setting capacity to 1.")
             capacity = 1

        for i in range(capacity):
            # Create unique zone IDs within the group
            zone_id = f"{self.group_id}-zone{i}"
            self.zones[zone_id] = NeuralZone(zone_id, self.config)

        logger.info(f"Created NeuralZoneGroup '{self.group_id}' with {len(self.zones)} zones.")

    def propagate_activations(self, external_stimuli: dict = None):
        """
        Updates the activation of all zones within the group for one time step.

        Applies a small random stimulus to each zone if no specific external
        stimuli are provided.

        Args:
            external_stimuli (dict, optional): A dictionary mapping zone identifiers
                                                to external stimulus tensors for this step.
                                                Defaults to None, applying random stimuli.
        """
        logger.debug(f"Propagating activations for group '{self.group_id}'")
        updated_count = 0
        for zone_id, zone in self.zones.items():
            # Determine stimulus: use provided external stimulus or a default random one
            if external_stimuli and zone_id in external_stimuli:
                stimulus = external_stimuli[zone_id]
            else:
                # Apply a small random stimulus if none provided
                stimulus = torch.rand(1, dtype=torch.float32) * 0.1 # Small random stimulus

            try:
                zone.update_activation(stimulus)
                updated_count += 1
            except Exception as e:
                 # Error is logged within update_activation
                 logger.error(f"Skipping activation update for zone '{zone_id}' due to error.")
                 continue # Skip to the next zone if update fails

        logger.debug(f"Completed activation propagation for {updated_count}/{len(self.zones)} zones in group '{self.group_id}'.")


    def connect_randomly(self, max_links_per_zone: int = None):
        """
        Creates random connections between zones within the group.

        Each zone attempts to connect to a random sample of other zones within
        the group, up to the maximum specified or the limit set in the ZoneConfig.

        Args:
            max_links_per_zone (int, optional): The maximum number of outgoing links
                                                to attempt creating for each zone.
                                                Defaults to config.MAX_CONNECTIONS.
        """
        if len(self.zones) < 2:
             logger.warning(f"Cannot establish random connections in group '{self.group_id}': requires at least 2 zones.")
             return

        max_links = max_links_per_zone if max_links_per_zone is not None else self.config.MAX_CONNECTIONS
        logger.info(f"Establishing random connections within group '{self.group_id}' (max_links_per_zone={max_links}).")

        zone_list = list(self.zones.values())
        connection_attempts = 0
        successful_connections = 0

        for source_zone in zone_list:
            # Potential targets are all zones except the source itself
            potential_targets = [z for z in zone_list if z != source_zone]
            if not potential_targets:
                continue # Should not happen if len(zones) >= 2

            # Determine how many connections to attempt for this zone
            # Cannot exceed max_links or the number of available targets
            num_targets_to_sample = min(max_links, len(potential_targets))

            # Ensure we don't exceed the zone's connection capacity
            num_targets_to_sample = min(num_targets_to_sample, self.config.MAX_CONNECTIONS - len(source_zone.connections))

            if num_targets_to_sample <= 0:
                continue # Already at max connections or no targets to sample

            # Randomly select target zones
            selected_targets = random.sample(potential_targets, num_targets_to_sample)

            for target_zone in selected_targets:
                connection_attempts += 1
                # link_zone handles checking connection limits again and logging
                if source_zone.link_zone(target_zone): # Default strength used
                    successful_connections += 1

        logger.info(f"Random connection process complete for group '{self.group_id}'. Attempted: {connection_attempts}, Succeeded: {successful_connections}.")

    def get_activation_states(self) -> dict:
        """
        Retrieves the current activation state of all zones in the group.

        Returns:
            dict: A dictionary mapping zone identifiers to their activation values (float).
        """
        return {zone_id: zone.activation.item() for zone_id, zone in self.zones.items()}

# Removed the original __main__ block as it was for interactive testing/demonstration.
# This functionality should be implemented in separate scripts or notebooks if needed.

