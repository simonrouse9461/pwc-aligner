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
                 positive_label: 0,
                 hc_thresholds=[0.9, 0.95]):
        super().__init__()
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.positive_label = positive_label
        self.hc_thresholds = hc_thresholds
        self.nsp = model
        self.train_acc = metrics.Accuracy()
        self.val_metrics = {
            'acc': metrics.Accuracy(),
            'pre': metrics.Precision(num_classes=2, average='none'),
            'rec': metrics.Recall(num_classes=2, average='none'),
            'cm': metrics.ConfusionMatrix(num_classes=2),
        }
        self.hc_val_metrics_dict = {
            threshold: {
                'acc': metrics.Accuracy(),
                'pre': metrics.Precision(num_classes=2, average='none'),
                'rec': metrics.Recall(num_classes=2, average='none'),
                'cm': metrics.ConfusionMatrix(num_classes=2),
            } for threshold in self.hc_thresholds
        }
    
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
        with torch.no_grad():
            output = self(**batch)
        losses, logits = output['loss'], output['logits']
        scores = F.softmax(logits, dim=1).cpu()
        labels = batch['labels'].cpu()
        false_idx = logits.argmax(dim=1) != labels
        for metric in self.val_metrics.values():
            metric(scores, labels)
        for threshold, hc_metrics in self.hc_val_metrics_dict.items():
            hc_idx = scores.max(dim=1).values > threshold
            try:
                for metric in hc_metrics.values():
                    metric(scores[hc_idx], labels[hc_idx])
            except:
                pass
        
        torch.cuda.empty_cache()
        return (np.array([batch['raw_sent1'], batch['raw_sent2'], labels.cpu().numpy()])
                .T[false_idx.cpu().numpy()]
                .tolist())
    
    def validation_epoch_end(self, outputs):
        def print_cm(a, p, r, cm, suffix=''):
            scores = f'accuracy: {a:.2f}, precision: {p:.2f}, recall: {r:.2f}'
            title = 'sanity check' if self.current_epoch == 0 else f'epoch {self.current_epoch}'
            self.print(f' {title}{suffix} '.center(len(scores), "="))
            self.print(pd.DataFrame(cm.long().cpu().numpy(), 
                                    columns=['pred: -', 'pred: +'], 
                                    index=['label: -', 'label: +']))
            self.print(scores)
            
        acc = self.val_metrics['acc'].compute()
        pre = self.val_metrics['pre'].compute()[self.positive_label]
        rec = self.val_metrics['rec'].compute()[self.positive_label]
        cm = self.val_metrics['cm'].compute()
        print_cm(acc, pre, rec, cm)
        for threshold, hc_metrics in self.hc_val_metrics_dict.items():
            try:
                hc_acc = hc_metrics['acc'].compute()
                hc_pre = hc_metrics['pre'].compute()[self.positive_label]
                hc_rec = hc_metrics['rec'].compute()[self.positive_label]
                hc_cm = hc_metrics['cm'].compute()
                print_cm(hc_acc, hc_pre, hc_rec, hc_cm, f' (> {threshold})')
            except:
                pass

        self.print(sum(outputs, []))
        
        for metric in self.val_metrics.values():
            metric.reset()
        for hc_metrics in self.hc_val_metrics_dict.values():
            for metric in hc_metrics.values():
                metric.reset()
        
    
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