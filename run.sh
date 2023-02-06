# !/bin/bash

archs=("vgg16" "resnet20" "resnet32" "resnet44" "resnet56" "resnet110" "densenet40")

echo "Training Models from Scratch"
for arch in ${archs[@]}
do
	batch_size=128 optimizer="adam" learning_rate=1e-3 decay_step="80,120,160,180"
	case $arch in
		"resnet20"|"resnet32")
			batch_size=32
			;;
		"densenet40")
			batch_size=64 optimizer="sgd" learning_rate=1e-1 decay_step="50,150"
			;;
		*)
	esac
	python train.py --arch $arch --batch-size $batch_size --optimizer $optimizer --learning-rate $learning_rate --decay-step $decay_step
done

echo "Pruning and Fine-tuning Models"
for arch in ${archs[@]}
do
	thresholds=(0.7 0.75 0.8)
	samples=100 batch_size=128
	case $arch in
		"vgg16")
			thresholds=(0.7 0.75 0.8 0.85 0.9 0.95)
			samples=50
			;;
		"resnet20"|"resnet32")
			batch_size=32
			;;
		"densenet40")
			thresholds=(0.6 0.65 0.7)
			;;
		*)
	esac
	for threshold in ${thresholds[@]}
	do
		python prune.py --arch $arch --threshold $threshold --samples $samples --batch-size $batch_size
	done
done
