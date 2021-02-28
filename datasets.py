from .imports import *


class DataMixer(IterableDataset):
    '''
    Utility to iterate over multiple datasets
    '''
    def __init__(self, *args):
        self.datasets = args
        
    def __iter__(self):
        for dataset in self.datasets:
            for data in dataset:
                yield data
    
    def __len__(self):
        return sum(map(len, self.datasets))
    
    
class CtrlRegMappingTrainingDataOld(IterableDataset):
    """
    Dataset class for loading old control regulation mapping training data.
    Note: For each positive mapping pair, a negative mapping pair with the same control 
    description will be randomly sampled.
    """
    def __init__(self, *,
                 file: str = 'aligner_data/control_reg_mappings.xlsx',
                 shuffle: bool = False, 
                 rollup_level: Optional[int] = None,
                 reg_filter: Optional[str] = None,
                 split_name: Optional[str] = None):
        """
        Args:
            file: 
                data file name
            shuffle: 
                shuffle the data loader or not
            rollup_level:
                the deepest level that the regulations can be broken down to, default to leaf nodes
            reg_filter: 
                filter regulations by substring
            split_name: 
                the data split name, corresponding to the suffix in the control data sheet names
        """
        super().__init__()
        regs_sheet = 'regulations' if rollup_level is None else f'regulations_level{rollup_level}'
        ctrls_sheet = 'controls' if split_name is None else f'controls_{split_name}'
        regs = (pd.read_excel(file, sheet_name=regs_sheet, header=None)
                  .drop_duplicates()
                  .dropna()
                  .reset_index(drop=True))
        ctrls = (pd.read_excel(file, sheet_name=ctrls_sheet, header=None)
                   .drop_duplicates()
                   .dropna()
                   .reset_index(drop=True))
        self.control_title, self.control_desc, self.obligations = ctrls[0], ctrls[1], ctrls[2]
        self.leaf_nodes, self.leaf_nodes_desc = regs[0], regs[1]
        self.shuffle = shuffle
        self.rollup_level = rollup_level
        self.reg_filter = reg_filter if reg_filter is not None else ''
        
        missing_regs = self.check_missing_regs()
        if len(missing_regs) > 0:
            msg = f'missing regs for {split_name} split:'
            print(''.center(len(msg), '='), file=sys.stderr)
            print(msg, file=sys.stderr)
            print(''.center(len(msg), '='), file=sys.stderr)
            print('\n'.join(missing_regs), file=sys.stderr)
        
    def __iter__(self):
        datalist = list(zip(self.control_desc, self.obligations.tolist()))
        if self.shuffle:
            random.shuffle(datalist)
        for control, obligation in datalist:
            matched_index = self._get_matched_leaf_node_index(obligation)
            matched = self.leaf_nodes_desc[matched_index].tolist()
            irrelvant = self.leaf_nodes_desc[~matched_index].tolist()
            for pos_reg in matched:
                neg_reg = random.choice(irrelvant)
                yield control, pos_reg, neg_reg

    @lru_cache(maxsize=1)
    def __len__(self):
        return sum(self._get_matched_leaf_node_index(obligation).sum() 
                   for obligation in self.obligations.tolist())
    
    def _reg_id_rollup(self, reg_id: str, level: Optional[int] = None) -> str:
        if level is None:
            return reg_id
        return ':'.join(reg_id.split(':')[:level+2])
    
    def _get_matched_leaf_node_index(self, obligation: str) -> pd.Series:
        """
        Expand obligation string into leaf node indices in boolean series format
        Args:
            obligation:
                the obligation string
        Returns:
            boolean series for leaf node indices
        """
        matched_index_list = [self.leaf_nodes.str.contains(self._reg_id_rollup(reg, self.rollup_level))
                              & self.leaf_nodes.str.contains(self.reg_filter)
                              for reg in obligation.split(',')]
        return reduce(lambda a, b: a | b, matched_index_list)
                
    def check_missing_regs(self) -> List[str]:
        """
        Print a list of missing regs which are referenced by at least one control.
        """
        leaf_node_ids = ','.join(set(self.leaf_nodes))
        obligation_set = set(sum([ids.split(',') for ids in self.obligations], []))
        return sorted(lnid for lnid in obligation_set if lnid not in leaf_node_ids)
    
    
class CtrlRegMappingTrainingData(IterableDataset):
    """
    Dataset class for loading new control regulation mapping training data.
    Note: The regulations in the new dataset are formated in level 3.
    """

    reg_id_col = 'Level 3 Citation'
    reg_desc_col = 'Compliance Requirement or T&C Description (From Metric Stream or T&C Inventory)'
    ctrl_desc_col = 'Scrubbed_Description'
    
    def __init__(self, *,
                 file: str = 'aligner_data/scrubbed_data_new.xlsx', 
                 shuffle: bool = False, 
                 reg_filter: Optional[str] = None,
                 split_slice: slice = slice(None)):
        """
        Args:
            file: 
                data file name
            shuffle: 
                shuffle the data loader or not
            reg_filter: 
                filter regulations by substring
            split_slice: 
                the data split slice
        """
        super().__init__()
        self.data = (pd.read_excel(file, sheet_name='Scrubbed_Data_CMU_10.28.20', header=0)
                     .drop_duplicates()
                     .dropna()
                     .reset_index(drop=True))[[self.reg_id_col, self.reg_desc_col, self.ctrl_desc_col]]
        if reg_filter is not None:
            self.data = self.data[self.data.apply(lambda x: reg_filter in x[self.reg_id_col], axis=1)]
        self.regs = self.data.groupby(self.reg_id_col).first()[[self.reg_desc_col]]
        self.mappings = self.data.groupby(self.ctrl_desc_col).apply(lambda g: pd.DataFrame(g[self.reg_id_col].to_list()))
        self.split = split_slice
        self.shuffle = shuffle
        
    def __iter__(self):
        idx_list = list(range(len(self)))
        if self.shuffle:
            random.shuffle(idx_list)
        for i in idx_list:
            yield self[i]

    def __getitem__(self, index):
        pos_reg, ctrl = self.data.iloc[self.split].iloc[index].loc[[self.reg_desc_col, self.ctrl_desc_col]]
        neg_regs = self.regs[~self.regs.index.isin(self.mappings.loc[ctrl][0])][self.reg_desc_col]
        neg_reg = random.choice(neg_regs.to_list())
        return ctrl, pos_reg, neg_reg
        
    def __len__(self):
        return len(self.data.iloc[self.split])
    

class MaskedLMBatchCollator(nn.Module):
    def __init__(self, tokenizer, max_len: int, mask_ratio: float, part: str = 'both'):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.part = part

    def forward(self, batch: List[Tuple[torch.Tensor, ...]]) -> Dict[str, torch.Tensor]:
        pairs_or_triples = list(zip(*batch))
        ctrl, reg = pairs_or_triples[0], pairs_or_triples[1]
        if self.part == 'both':
            sequence = list(ctrl + reg)
        elif self.part == 'ctrl':
            sequence = list(ctrl)
        elif self.part == 'reg':
            sequence = list(reg)
        else:
            raise Exception(f'Unknow part {self.part}')
        encoding = self.tokenizer(sequence, 
                                  return_tensors="pt", 
                                  max_length=self.max_len,
                                  padding=True, 
                                  truncation=True,
                                  add_special_tokens=True,
                                  return_special_tokens_mask=True,
                                  return_attention_mask=True)
        labels = encoding.input_ids.clone()
        mask_token_mask = (torch.rand_like(encoding.input_ids, dtype=float)
                                .gt(self.mask_ratio)
                                .logical_or(encoding.special_tokens_mask)
                                .logical_not())
        encoding.input_ids[mask_token_mask] = self.tokenizer.mask_token_id
        return {'input_ids': encoding.input_ids, 
                'attention_mask': encoding.attention_mask,
                'mask_token_mask': mask_token_mask,
                'labels': labels}
    

class CtrlRegMappingBatchCollator(nn.Module):
    """
    The collate function that merges a list of data into a single batch
    """
    
    def __init__(self, tokenizer, max_len: int, train: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train = train

    def forward(self, batch: List[Tuple[torch.Tensor, ...]]) -> Dict[str, torch.Tensor]:
        pairs_or_triples = list(zip(*batch))
        ctrl, pos_reg = pairs_or_triples[0], pairs_or_triples[1]
        if self.train:
            neg_reg = pairs_or_triples[2]
            sent1, sent2 = ctrl * 2, pos_reg + neg_reg
            labels = torch.cat([torch.zeros(len(ctrl)), 
                                torch.ones(len(ctrl))]).long()
        else:
            sent1, sent2 = ctrl, pos_reg
            labels = None
        encoding = self.tokenizer(sent1, sent2,
                                  return_tensors="pt", 
                                  max_length=self.max_len,
                                  padding=True, 
                                  truncation=True,
                                  add_special_tokens=True,
                                  return_token_type_ids=True,
                                  return_attention_mask=True)
        return {'input_ids': encoding.input_ids, 
                'token_type_ids': encoding.token_type_ids, 
                'attention_mask': encoding.attention_mask,
                'labels': labels,
                'raw_sent1': sent1,
                'raw_sent2': sent2}
    

class MaskedLMDataModule(pl.LightningDataModule):
    def __init__(self, *,
                 tokenizer: PreTrainedTokenizer, 
                 batch_size: int,
                 max_len: int,
                 part: str = 'both',
                 mask_ratio: float = 0.15):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data = DataMixer(
            CtrlRegMappingTrainingData(shuffle=True),
            CtrlRegMappingTrainingDataOld(shuffle=True, rollup_level=3)
        )
        self.batch_collator = MaskedLMBatchCollator(tokenizer=self.tokenizer,
                                                    max_len=max_len,
                                                    mask_ratio=mask_ratio,
                                                    part=part)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.data,
                          batch_size=self.batch_size,
                          collate_fn=self.batch_collator,
                          shuffle=False,
                          num_workers=1) # ensure num_workers=1 to avoid duplicated entries in each epoch
    
    
class CtrlRegMappingNSPDataModule(pl.LightningDataModule):
    """
    Module for training, validation, and testing data provisioning
    """
    def __init__(self, *,
                 tokenizer: PreTrainedTokenizer, 
                 batch_size: int,
                 max_len: int,
                 experiment: str = 'all-new',
                 train_batch_size: Optional[int] = None,
                 val_batch_size: Optional[int] = None,
                 test_batch_size: Optional[int] = None):
        super().__init__()
        self.train_batch_size = train_batch_size or batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.tokenizer = tokenizer
        self.train_batch_collator = CtrlRegMappingBatchCollator(tokenizer=self.tokenizer,
                                                                max_len=max_len,
                                                                train=True)
        self.test_batch_collator = CtrlRegMappingBatchCollator(tokenizer=self.tokenizer,
                                                               max_len=max_len,
                                                               train=False)
        if experiment == 'all-new':
            self.train_data = CtrlRegMappingTrainingData(shuffle=True, split_slice=slice(None, 935))
            self.val_data = CtrlRegMappingTrainingData(shuffle=False, split_slice=slice(935, None))
        if experiment == 'all-old':
            self.train_data = CtrlRegMappingTrainingDataOld(shuffle=True, rollup_level=None, split_name='train')
            self.val_data = CtrlRegMappingTrainingDataOld(shuffle=False, rollup_level=None, split_name='val')
        if experiment == 'cfr-train_old-val_new':
            self.train_data = CtrlRegMappingTrainingDataOld(shuffle=True, rollup_level=3, reg_filter='CFR')
            self.val_data = CtrlRegMappingTrainingData(shuffle=False, reg_filter='CFR')
        if experiment == 'usc-train_new-val_old':
            self.train_data = CtrlRegMappingTrainingData(shuffle=True, reg_filter='USC')
            self.val_data = CtrlRegMappingTrainingDataOld(shuffle=False, rollup_level=3, reg_filter='USC')
        if experiment == 'cfr-mix':
            self.train_data = DataMixer(
                CtrlRegMappingTrainingData(shuffle=True, reg_filter='CFR', split_slice=slice(None, 540)),
                CtrlRegMappingTrainingDataOld(shuffle=True, rollup_level=3, reg_filter='CFR', split_name='train')
            )
            self.val_data = DataMixer(
                CtrlRegMappingTrainingData(shuffle=False, reg_filter='CFR', split_slice=slice(540, None)),
                CtrlRegMappingTrainingDataOld(shuffle=False, rollup_level=3, reg_filter='CFR', split_name='val')
            )
        if experiment == 'all-mix':
            self.train_data = DataMixer(
                CtrlRegMappingTrainingData(shuffle=True, split_slice=slice(None, 900)),
                CtrlRegMappingTrainingDataOld(shuffle=True, rollup_level=3, split_name='train')
            )
            self.val_data = DataMixer(
                CtrlRegMappingTrainingData(shuffle=False, split_slice=slice(900, None)),
                CtrlRegMappingTrainingDataOld(shuffle=False, rollup_level=3, split_name='val')
            )
        self.test_data = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_data,
                          batch_size=self.train_batch_size,
                          collate_fn=self.train_batch_collator,
                          shuffle=False,
                          num_workers=1) # ensure num_workers=1 to avoid duplicated entries in each epoch

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_data,
                          batch_size=self.val_batch_size,
                          collate_fn=self.train_batch_collator,
                          shuffle=False,
                          num_workers=1) # ensure num_workers=1 to avoid duplicated entries in each epoch
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_data,
                          batch_size=self.test_batch_size,
                          collate_fn=self.test_batch_collator,
                          shuffle=False,
                          num_workers=1) # ensure num_workers=1 to avoid duplicated entries in each epoch
    
    def get_top_k_predictions_from_test_results(self, scores: torch.Tensor, k: int = 5) -> Dict[str, List[str]]:
        top_k_idx = (scores.reshape([self.test_data.n_ctrl, self.test_data.n_reg])
                           .argsort(descending=True)[:, :k]
                           .numpy())
        predicted_mapping = self.test_data.leaf_nodes.to_numpy()[top_k_idx].tolist()
        controls = self.test_data.control_title.tolist()
        return dict(zip(controls, predicted_mapping))