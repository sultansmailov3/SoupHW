# Soup QA strategy

Start with a valid offline plan and a tiny SFT training control. Highest priority is a successful plan or training run that silently uses a different setting. Generate bounded config mutations and record both streams, exit status and input for every case. Check duplicate keys against the effective config and compare actual optimizer steps against a clean control.

Run the same seed twice in fresh processes on the same GPU with an identical local model and dataset. Compare the loss curve by step, the final adapter tensors and outcome metrics. Preserve all results. Large models, throughput, other backends and other GPUs are outside scope.
