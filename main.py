import argparse
from controller import Controller
from work_states import SystemWorkStates

work_modes = {
    'image': SystemWorkStates.image,
    'webcam': SystemWorkStates.webcam,
    'video': SystemWorkStates.video
}


def main():
    parser = argparse.ArgumentParser(description='take a picture')
    parser.add_argument('-mode', '-n', default='unknown', type=str, help='input the name of the recording person')
    args = parser.parse_args()
    controller = Controller()
    controller.run(work_modes[args.mode])


if __name__ == '__main__':
    main()
