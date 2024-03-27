import configparser
import potentials
from util import error


# map func setting to Python function and parameters
potential_funcs = {
    'harmonic'        : (potentials.Harmonic1D  , ['k' ,'x0']    ),
    'double-well'     : (potentials.DW1D        , ['a' ,'b' ,'c']),
    'double-well-sym' : (potentials.SymDW1D     , ['x0','h' ]    ),
    'double-well-asym': (potentials.ASymDW1D    , ['x0','v' ,'h']),
    'mueller-brown'   : (potentials.MuellerBrown, []             ),
    }


class Config:
    def __init__(self, filename, overrides=[]):
        # read the input file
        inp = configparser.ConfigParser(
            delimiters = '=',
            comment_prefixes = '#',
            inline_comment_prefixes = '#',
            strict = True)
        inp.read(filename)
        self.inputfile = filename
        self.input = inp
        # apply overrides
        self.update(overrides)
        # store derived information
        kB = self.getfloat('trajectory', 'kB')
        T  = self.getfloat('trajectory', 'temp')
        self.beta = 1.0/(kB*T)
        # instantiate the potential function object
        func = self.get('potential', 'func')
        pyfunc, paramlist = potential_funcs[func]
        paramdict = { param : self.getfloat('potential', param) for param in paramlist }
        self.pot = pyfunc(**paramdict)

    def update(self, params):
        for p in params:
            key,value = p.split('=', maxsplit=1)
            section,setting = key.split('.', maxsplit=1)
            self.guard(section, setting)
            self.input[section][setting] = value

    def guard(self, section, setting):
        if not self.input.has_section(section):
            error(f"section '{section}' missing from input file")
        if not self.input.has_option(section, setting):
            error(f"setting '{setting}' missing from input file section '{section}'")

    def get(self, section, setting):
        self.guard(section, setting)
        return self.input.get(section, setting)

    def getint(self, section, setting):
        self.guard(section, setting)
        try:
            return self.input.getint(section, setting)
        except ValueError:
            value = self.input.get(section, setting)
            error(f"setting '{section}.{setting}' must be an integer, not '{value}'")

    def getfloat(self, section, setting):
        self.guard(section, setting)
        try:
            return self.input.getfloat(section, setting)
        except ValueError:
            value = self.input.get(section, setting)
            error(f"setting '{section}.{setting}' must be a floating-point number, not '{value}'")

