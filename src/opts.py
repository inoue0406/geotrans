import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='../data',
        help='dataset path')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='../run/saved_models/VisionTransformers/',
        help='path of pretrained models')
    parser.add_argument(
        '--embed_dim',
        default=256,
        type=int,
        help='Dimensionality of input and attention feature vectors')
    parser.add_argument(
        '--hidden_dim',
        default=512,
        type=int,
        help='Dimensionality of hidden layer in feed-forward network')
    parser.add_argument(
        '--num_heads',
        default=8,
        type=int,
        help='Number of heads to use in the Multi-Head Attention block.')
    parser.add_argument(
        '--num_layers',
        default=6,
        type=int,
        help='Number of layers to use in the Transformer')
    parser.add_argument(
        '--patch_size',
        default=4,
        type=int,
        help='Number of pixels that the patches have per dimension')
    parser.add_argument(
        '--num_channels',
        default=3,
        type=int,
        help='Number of channels of the input (3 for RGB)')
    parser.add_argument(
        '--num_patches',
        default=64,
        type=int,
        help='Maximum number of patches an image can have')
    parser.add_argument(
        '--num_classes',
        default=10,
        type=int,
        help='Number of classes to predict')
    parser.add_argument(
        '--dropout',
        default=0.2,
        type=float,
        help='Dropout Ratio')
    parser.add_argument(
        '--lr',
        default=3e-4,
        type=float,
        help='Learning Rate')
    
    args = parser.parse_args()

    return args
