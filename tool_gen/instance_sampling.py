from __future__ import annotations

import random
from sweagent.run.batch_instances import (
	AbstractInstanceSource,
	CustomSingleInstanceSource,
	CustomBatchInstanceSource,
)


def sample_single_instance(instances_config: AbstractInstanceSource, rng: random.Random) -> AbstractInstanceSource:
	"""Return a single-instance source sampled at random from available configs."""
	all_configs = instances_config.get_instance_configs()
	if not all_configs:
		return instances_config
	rng.shuffle(all_configs)
	selected = rng.choice(all_configs)
	return CustomSingleInstanceSource(instance=selected)


def sample_batch_instances(
	instances_config: AbstractInstanceSource,
	rng: random.Random,
	batch_size: int,
) -> AbstractInstanceSource:
	"""Return a batch-instance source sampled uniformly at random."""
	all_configs = instances_config.get_instance_configs()
	if not all_configs:
		return instances_config
	batch_size = min(batch_size, len(all_configs))
	rng.shuffle(all_configs)
	instances = rng.sample(all_configs, batch_size)
	return CustomBatchInstanceSource(instances=instances)
