import sagemaker
from sagemaker import get_execution_role
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--entry_point', type=str)
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--input_s3', type=str)
    parser.add_argument('--version', type=str)


    args = parser.parse_args()

    entry_point = args.entry_point
    src_dir = args.src_dir
    inputs = args.input_s3
    version = args.version

    from sagemaker.tensorflow import TensorFlow

    mnist_estimator = TensorFlow(entry_point=entry_point,
                                 role=get_execution_role(),
                                 source_dir=src_dir,
                                 framework_version='1.12.0',
                                 training_steps=100,
                                 evaluation_steps=10,
                                 train_instance_count=1,
                                 tags={"version": version},
                                 train_instance_type='ml.m4.xlarge')

    mnist_estimator.fit(inputs)
