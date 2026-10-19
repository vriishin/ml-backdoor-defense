## Basic ABL (with ResNet)

This process performs the default ABL steps with no pruning of the models in between.

Perform backdoor isolation to divide the dataset into 'clean' samples and a small set of 'poisoned' samples.

```bash
$ python backdoor_isolation.py --model_name "ResNet34"
```

Perform backdoor unlearning using the clean and poisoned samples obtained from the backdoor isolation step.

```bash
$ python backdoor_unlearning.py --model_name "ResNet34"
```
## ABL + ANP Optimal (with ResNet)

This process pertains to the 'ideal' path for mixing ANP with ABL. Here, mask optimization and pruning of the models is performed AFTER fine tuning to prevent backdoor relearning.

Perform backdoor isolation to divide the dataset into 'clean' samples and a small set of 'poisoned' samples (with the norm layer set to NoisyBatchNorm2d for ANP).

```bash
$ python backdoor_isolation.py --model_name "ResNet34" --norm_layer "NoisyBatchNorm2d"
```
Perform fine tuning (unlearning epochs is set to 0 to only perform fine tuning) using the tuned model from the backdoor isolation step. 

```bash
$ python backdoor_unlearning.py --model_name "ResNet34" --norm_layer "NoisyBatchNorm2d" --unlearning_epochs 0
```
Perform mask optimization over the model using the clean samples obtained from the backdoor isolation step. To use a custom dataset emanating form backdoor isolation, add the parameter --mask_opt_data_type "custom". The parameter with_finetuned_model indicates that the mask optimization is using the fine tuned model instead of the model emanating directly from backdoor isolation.

```bash
$ python mask_optimization.py --model_name "ResNet34" --norm_layer "NoisyBatchNorm2d" --with_finetuned_model True
```

Perform neuron pruning on the model based on the masks obtained from the mask optimization step.

```bash
$ python neuron_pruning.py --model_name "ResNet34" --norm_layer "NoisyBatchNorm2d"
```

Perform backdoor unlearning (fine tuning epochs is set to 0 to only perform backdoor unlearning) on the pruned model using the clean and poisoned samples from the backdoor isolation step. The parameter --with_pruned_model True sets the target model to the pruned model.

```bash
$ python backdoor_unlearning.py --model_name "ResNet34" --norm_layer "NoisyBatchNorm2d" --with_pruned_model True --finetuning_epochs 0
```

## ABL + ANP (with ResNet)

This process pertains to our intentded path for mixing ANP with ABL. Here, pruning is performed prior to fine tuning and backdoor unlearning. The goal is to outperform ABL while matching the performance of the 'optimal' process detailed above as closely as possible.

Perform backdoor isolation to divide the dataset into 'clean' samples and a small set of 'poisoned' samples (with the norm layer set to NoisyBatchNorm2d for ANP).

```bash
$ python backdoor_isolation.py --model_name "ResNet34" --norm_layer "NoisyBatchNorm2d"
```

Perform mask optimization over the model using the clean samples obtained from the backdoor isolation step. To use a custom dataset emanating form backdoor isolation, add the parameter --mask_opt_data_type "custom".

```bash
$ python mask_optimization.py --model_name "ResNet34" --norm_layer "NoisyBatchNorm2d"
```

Perform neuron pruning on the model based on the masks obtained from the mask optimization step.

```bash
$ python neuron_pruning.py --model_name "ResNet34" --norm_layer "NoisyBatchNorm2d"
```

Perform fine tuning and backdoor unlearning on the pruned model using the clean and poisoned samples from the backdoor isolation step.  To use a custom dataset emanating form backdoor isolation, add the parameter --finetuning_data_type "custom". The parameter --with_pruned_model True sets the target model to the pruned model.

```bash
$ python backdoor_unlearning.py --model_name "ResNet34" --norm_layer "NoisyBatchNorm2d" --with_pruned_model True
```
