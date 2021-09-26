import argparse
import main_module

def main():
	parser = argparse.ArgumentParser(description='Image classification model')
	
	parser.add_argument('-ep','--epochs', metavar="", type=int, required=True, help='Number of epochs')
	
	parser.add_argument('-m','--model', metavar="", type=str, required=True,
		choices=("r18","r34","r50"), help='Classification model')
	parser.add_argument('-b','--batch_size', metavar="", type=int, required=True,
		choices=(16, 32, 64), help='train batch size')

	args = parser.parse_args()

	result = main_module.main(args.epochs, args.model, args.batch_size)
	return result

if __name__ == '__main__':
	main()