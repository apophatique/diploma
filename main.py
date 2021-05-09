import argparse
from controller import Controller

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='webcam', type=str)
parser.add_argument('-auditorium', type=str)
parser.add_argument('-date', type=str)
args = parser.parse_args()


def main():
    controller = Controller()
    controller.run(
        args.mode
    )


if __name__ == '__main__':
    main()
