""" The scope to store global variables in ansor """

class AutoschedulerGlobalScope(object):
    def __init__(self):
        self.topi_in_compute_rewrite_mode = False

GLOBAL_SCOPE = AutoschedulerGlobalScope()

