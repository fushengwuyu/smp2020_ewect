# author: sunshine
# datetime:2021/7/23 上午10:17
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from src.model import SMPNet
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loaders, tokenizer, id2label):

        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")

        self.model = SMPNet(args, len(id2label))
        self.id2label = id2label
        self.model.to(self.device)
        if args.train_mode == "eval":
            self.resume()

        self.train_dataloader, self.dev_usual_dataloader, self.dev_virus_dataloader = data_loaders

        # 设置优化器，优化策略
        train_steps = (len(self.train_dataloader) / args.batch_size) * args.epoch_num
        self.optimizer, self.schedule = self.set_optimizer(args=args,
                                                           model=self.model,
                                                           train_steps=train_steps)

        self.loss_fc = torch.nn.CrossEntropyLoss()

    def set_optimizer(self, args, model, train_steps=None):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # optimizer, num_warmup_steps, num_training_steps
        schedule = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=train_steps
        )
        return optimizer, schedule

    def train(self):

        best_f1 = 0.0
        self.model.train()
        step_gap = 5
        step_eval = 10
        for epoch in range(int(self.args.epoch_num)):
            for step, batch in tqdm(enumerate(self.train_dataloader)):

                loss = self.forward(batch, is_eval=False)
                if step % step_gap == 0:
                    info = "step {} / {} of epoch {}, train/loss: {}"
                    print(info.format(step, len(self.train_dataloader) / self.args.batch_size, epoch, loss.item()))

                if step % step_eval == 0:

                    p, r, f1 = self.evaluate()
                    print("p: {}, r: {}, f1: {}".format(p, r, f1))
                    if f1 >= best_f1:
                        best_f1 = f1

                        # 保存模型
                        self.save()

    def forward(self, batch, is_eval=False):

        if not is_eval:
            batch = tuple(t.to(self.device) for t in batch)
            label1, label2 = batch[6:]
            self.optimizer.zero_grad()

            out1, out2 = self.model(x1=batch[:3], x2=batch[3:6])

            loss1 = self.loss_fc(out1, label1)
            loss2 = self.loss_fc(out2, label2)
            loss = loss1 + loss2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.warmup_proportion)
            self.optimizer.step()
            self.schedule.step()

            return loss
        else:
            x = tuple(t.to(self.device) for t in batch[:-2])
            label, task = batch[-2], batch[-1]
            if task == 'usual':
                out = self.model(x1=x)
            else:
                out = self.model(x2=x)
            out = out[0].softmax(dim=-1)
            pred = torch.argmax(out, dim=-1).cpu().numpy()
            correct = sum(label == pred)
            return correct, len(pred)

    def resume(self):
        resume_model_file = self.args.output + "/pytorch_model.bin"
        logging.info("=> loading checkpoint '{}'".format(resume_model_file))
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def save(self):
        logger.info("** ** * Saving fine-tuned model ** ** * ")
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Only save the model it-self
        output_model_file = self.args.output + "/pytorch_model.bin"
        torch.save(model_to_save.state_dict(), str(output_model_file))

    def evaluate(self):
        """验证
        """
        self.model.eval()
        correct_usual, correct_virus = 0, 0
        num_usual, num_virus = 0, 0
        with torch.no_grad():
            for batch in tqdm(self.dev_usual_dataloader):
                correct, n = self.forward(batch=batch, is_eval=True)
                correct_usual += correct
                num_usual += n

            for batch in tqdm(self.dev_virus_dataloader):
                correct, n = self.forward(batch=batch, is_eval=True)
                correct_virus += correct
                num_virus += n

        self.model.train()
        return (correct_usual + correct_virus) / (
                num_usual + num_virus), correct_usual / num_usual, correct_virus / num_virus


if __name__ == '__main__':
    def f1(a, b, c):
        print(a, b, c)


    f1(*[1, 2, 3])
