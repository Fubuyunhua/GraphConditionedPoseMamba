"""Opt-in all-parameter layerwise LR groups for W128/D20 detector adaptation."""
def layerwise_lr_groups(model, groups, args):
    if not getattr(args, 'layerwise_finetune', False):
        return groups
    if not args.finetune or args.gt_2d or getattr(args,'selective_last_block_head',False):
        raise ValueError('Registered layerwise fine-tune requires full-parameter detector adaptation')
    if len(model.blocks)!=20 or model.embed_dim!=128:
        raise ValueError('Registered layerwise recipe requires W128/D20')
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Layerwise recipe must not freeze parameters')
    names={id(p):n for n,p in model.named_parameters()}
    def bucket(name):
        if name.startswith('head.'):return 'head'
        if name.startswith('blocks.'):
            return 'early' if int(name.split('.')[1])<10 else 'late'
        return 'input'
    rates={'input':args.learning_rate,'early':args.learning_rate,
           'late':args.late_learning_rate,'head':args.head_learning_rate}
    result=[]
    for group in groups:
        for label,lr in rates.items():
            params=[p for p in group['params'] if bucket(names[id(p)])==label]
            if params:result.append({**group,'params':params,'lr':float(lr),'group_name':label+'_'+group['group_name']})
    actual=[id(p) for g in result for p in g['params']]
    expected={id(p) for p in model.parameters()}
    if set(actual)!=expected or len(actual)!=len(expected):raise ValueError('Optimizer parameter coverage mismatch')
    return result
