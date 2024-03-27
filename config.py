import configparser
import potentials


class Config:
    def __init__(self, filename):
        inp = configparser.ConfigParser(
            delimiters = '=',
            comment_prefixes = '#',
            inline_comment_prefixes = '#',
            strict = True)
        inp.read(filename)
        self.inputfile = filename
        self.input = inp
        kB = inp.getfloat('trajectory', 'kB')
        T  = inp.getfloat('trajectory', 'temp')
        self.beta = 1.0/(kB*T)
        self.pot = pot_from_config(inp)


# map func setting to Python function and parameters
potential_funcs = {
    'harmonic'        : (potentials.Harmonic1D  , ['k' ,'x0']    ),
    'double-well'     : (potentials.DW1D        , ['a' ,'b' ,'c']),
    'double-well-sym' : (potentials.SymDW1D     , ['x0','h' ]    ),
    'double-well-asym': (potentials.ASymDW1D    , ['x0','v' ,'h']),
    'mueller-brown'   : (potentials.MuellerBrown, []             ),
    }

def pot_from_config(inp):
    func = inp.get('potential', 'func')
    pyfunc, paramlist = potential_funcs[func]
    paramdict = { param : inp.getfloat('potential', param) for param in paramlist }
    pot = pyfunc(**paramdict)
    return pot

