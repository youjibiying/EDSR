import torch

import utility as utility

import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
args.cpu=args.cpu if torch.cuda.is_available() else True

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()


    else:
        if checkpoint.ok:
            loader = data.Data(args) # 得data_lader
            _model = model.Model(args, checkpoint) #获得模型
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint) # 构造’训练测试类‘
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
