import sagemaker
from sagemaker import get_execution_role
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--entry_point', type=str)
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--input_s3', type=str)

    args = parser.parse_args()

    entry_point = args.entry_point
    src_dir = args.src_dir
    inputs = args.input_s3

    from sagemaker.tensorflow import TensorFlow

    mnist_estimator = TensorFlow(entry_point=entry_point,
                                 role=get_execution_role(),
                                 source_dir=src_dir,
                                 framework_version='1.12.0',
                                 training_steps=100,
                                 evaluation_steps=10,
                                 train_instance_count=1,
                                 train_instance_type='local')

    mnist_estimator.fit(inputs)
