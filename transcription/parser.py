import argparse

def parse_args(): 
    parser = argparse.ArgumentParser(description="Initialize parameters")
    
    parser.add_argument('--hf_token', nargs='?', default='your_hugging_face_token',
                        help='load in hugging face token')
    
    return parser.parse_args()