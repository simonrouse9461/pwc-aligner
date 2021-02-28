from .imports import *


class MaskedLMPretrainingModel(pl.LightningModule):
    def __init__(self, *,
                 model: PreTrainedModel, 
                 learning_rate: float = 1e-6,
                 lr_decay_rate: float = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.mlm = model
        self.acc = metrics.Accuracy()

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        outputs = self.mlm(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
        return outputs

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.001)
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.lr_decay_rate)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss, logits = output['loss'], output['logits']
        mask = batch['mask_token_mask']
        labels = batch['labels'][mask]
        preds = logits.argmax(dim=2)[mask]
        self.acc(preds, labels)
        self.log('acc', self.acc, 
                 on_step=True, 
                 on_epoch=True, 
                 logger=True, 
                 prog_bar=True)
        
        torch.cuda.empty_cache()
        return output['loss']

    def training_epoch_end(self, outputs):
        self.print(np.array(list(map(lambda o: o['loss'].cpu().numpy(), outputs))).mean())
        
        
class CtrlRegAlignerModel(pl.LightningModule):
    def __init__(self, *,
                 model: PreTrainedModel, 
                 learning_rate: float = 1e-6,
                 lr_decay_rate: float = 0.9,
                 hc_threshold=0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.hc_threshold = hc_threshold
        self.nsp = model
        self.train_acc = metrics.Accuracy()
        self.val_acc = metrics.Accuracy()
        self.val_prec = metrics.Precision()
        self.val_rec = metrics.Recall()
        self.val_cm = metrics.ConfusionMatrix(num_classes=2)
        self.val_acc_hc = metrics.Accuracy()
        self.val_prec_hc = metrics.Precision()
        self.val_rec_hc = metrics.Recall()
        self.val_cm_hc = metrics.ConfusionMatrix(num_classes=2)
    
    def forward(self, 
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.nsp(input_ids=input_ids, 
                           token_type_ids=token_type_ids, 
                           attention_mask=attention_mask,
                           labels=labels)
        return outputs
    
    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.001)
        self.scheduler = ExponentialLR(self.optimizer, gamma=self.lr_decay_rate)
        return [self.optimizer], [self.scheduler]
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        output = self(**batch)
        losses, logits = output['loss'], output['logits']
        loss = losses.mean()
        preds = logits.argmax(dim=1)
        self.train_acc(preds, batch['labels'])
        self.log('train_acc', self.train_acc, 
                 on_step=True, 
                 on_epoch=False, 
                 logger=True, 
                 prog_bar=True)
        self.log('lr', self.scheduler.get_last_lr()[0], prog_bar=True)
        
        torch.cuda.empty_cache()
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        output = self(**batch)
        losses, logits = output['loss'], output['logits']
        scores = F.softmax(logits, dim=1)
        labels = batch['labels']
        false_idx = logits.argmax(dim=1) != labels
        hc_idx = scores.max(dim=1).values > self.hc_threshold
        scores_hc = scores[hc_idx]
        labels_hc = labels[hc_idx]
        self.val_acc(scores, labels)
        self.val_prec(scores, labels)
        self.val_rec(scores, labels)
        self.val_cm(scores, labels)
        try:
            self.val_acc_hc(scores_hc, labels_hc)
            self.val_prec_hc(scores_hc, labels_hc)
            self.val_rec_hc(scores_hc, labels_hc)
            self.val_cm_hc(scores_hc, labels_hc)
        except:
            pass
        
        self.log('val_acc', self.val_acc, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True, 
                 prog_bar=True)
        self.log('val_prec', self.val_prec, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True, 
                 prog_bar=True)
        self.log('val_rec', self.val_rec, 
                 on_step=False, 
                 on_epoch=True, 
                 logger=True, 
                 prog_bar=True)
        
        torch.cuda.empty_cache()
        return np.array([batch['raw_sent1'], batch['raw_sent2'], labels.cpu().numpy()]).T[false_idx.cpu().numpy()].tolist()
    
    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()
        prec = self.val_prec.compute()
        rec = self.val_rec.compute()
        confusion = self.val_cm.compute()
        try:
            acc_hc = self.val_acc_hc.compute()
            prec_hc = self.val_prec_hc.compute()
            rec_hc = self.val_rec_hc.compute()
            confusion_hc = self.val_cm_hc.compute()
        except:
            pass
        
        def print_cm(a, p, r, cm, suffix=''):
            scores = f'accuracy: {a:.2f}, precision: {p:.2f}, recall: {r:.2f}'
            title = 'sanity check' if self.current_epoch == 0 else f'epoch {self.current_epoch}'
            self.print(f' {title}{suffix} '.center(len(scores), "="))
            self.print(pd.DataFrame(cm.long().cpu().numpy(), 
                                    columns=['pred: -', 'pred: +'], 
                                    index=['label: -', 'label: +']))
            self.print(scores)
            
        print_cm(acc, prec, rec, confusion)
        try:
            print_cm(acc_hc, prec_hc, rec_hc, confusion_hc, f' (> {self.hc_threshold})')
        except:
            pass
        self.print(sum(outputs, []))
    
    def test_step(self, batch: torch.Tensor, batch_idx: int):
        output = self(**batch)
        loss, logits = output['loss'], output['logits']
        labels = batch['labels'][mask]
        preds = logits.argmax(dim=2)[mask]

        torch.cuda.empty_cache()
        return result
    
    def test_epoch_end(self, results):
        torch.save(torch.tensor(sum(results.score, [])), 'predictions.pt')
        return results