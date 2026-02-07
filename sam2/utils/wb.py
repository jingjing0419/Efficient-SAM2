

class activation_hook:
    def __init__(self, module, out_act = True):
        if out_act:
            self.hook = module.register_forward_hook(self.hook_fn_output)
        else:
            self.hook = module.register_forward_hook(self.hook_fn_input)
            
    def hook_fn_output(self, module, input, output):
        self.feature = output
    def hook_fn_input(self, module, input, output):
        print('input')
        self.feature = input

    def remove(self):
        self.hook.remove()