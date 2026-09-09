"""Opt-in last-block/head fine-tuning; no changes to default training."""
def configure_selective(model, args):
    if not getattr(args, 'selective_last_block_head', False):
        return
    if not args.finetune or args.partial_train:
        raise ValueError('Selective mode requires finetune and no partial_train')
    if not hasattr(model, 'blocks') or len(model.blocks) != 16 or not hasattr(model, 'head'):
        raise ValueError('Selective mode requires the registered W256/D16 model')
    count = int(getattr(args, 'selective_train_blocks', 1))
    if count not in (1, 2):
        raise ValueError('Registered selective modes allow only one or two final blocks')
    model._selective_block_start = len(model.blocks) - count
    prefixes = tuple(f'blocks.{i}.' for i in range(model._selective_block_start, len(model.blocks))) + ('head.',)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    model._selective_finetune = True
    model._selective_frozen_names = {
        n for n, p in model.named_parameters() if not p.requires_grad
    }


def selective_train_mode(model):
    if not getattr(model, '_selective_finetune', False):
        return
    model.eval()
    for block in model.blocks[model._selective_block_start:]:
        block.train()
    model.head.train()


def selective_lr_groups(model, groups, args):
    if not getattr(args, 'selective_last_block_head', False):
        return groups
    names = {id(p): n for n, p in model.named_parameters()}
    result = []
    for group in groups:
        for label, prefix, lr in (
            ('last_block', tuple(f'blocks.{i}.' for i in range(model._selective_block_start, len(model.blocks))), args.learning_rate),
            ('head', 'head.', args.head_learning_rate),
        ):
            parameters = [p for p in group['params'] if names[id(p)].startswith(prefix)]
            if parameters:
                result.append({**group, 'params': parameters, 'lr': float(lr),
                               'group_name': label + '_' + group['group_name']})
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for g in result for p in g['params']]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Selective optimizer coverage mismatch')
    return result
