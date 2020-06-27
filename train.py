from argparse import ArgumentParser
from torch.utils.data import DataLoader

from opts import Opts
from metrics import acc
from data import get_train_val_test_dataset
from trainer import Trainer


def main(opt):
    trainer = Trainer(opt)
    train_dataset, validate_dataset, test_dataset = get_train_val_test_dataset(opt)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,\
                              num_workers=opt.num_workers, drop_last=opt.drop_last)
    validate_loader = DataLoader(validate_dataset, batch_size=opt.batch_size,\
                              num_workers=opt.num_workers, drop_last=opt.drop_last)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)

    trainer.train(train_loader, validate_loader, test_loader)


if __name__ == '__main__':
    parser = ArgumentParser('DBS')
    parser.add_argument('--cfg', type=str, default='base.yml')
    arg = parser.parse_args()

    opt = Opts(arg.cfg)
    main(opt)