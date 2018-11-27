import random as rand

# Library for producing data

#region HELPER FUNCTIONS

def try_or_return(maybe_func):
    """
    Tries to call maybe_func(), otherwise returns maybe_func
    """
    try:
        return maybe_func()
    except:
        return maybe_func

#endregion

#region PRODUCER DEFINITIONS
class AbstractProducer:
    """
    Any AbstractProducer can be "then-ed" with a procedure.
    Say:  P() -> x
    Then: P().then(f) -> f(x)
    """

    def __init__(self, producer):
        self.producer = producer

    def __call__(self):
        return self.producer()

    def then(self, procedure):
        return AbstractProducer(lambda : procedure(self()))

class InstanceProducer(AbstractProducer):
    """
    Produces an instance of a class
    """

    def __init__(self, cls, *fakers, **faker_kwargs):
        self.cls = cls

        faker_lst = list(fakers)
        self.args_faker = ListProducer(faker_lst, len=len(faker_lst))

        self.kwargs_faker = DictProducer(**faker_kwargs)

    def __call__(self):
        return self.cls(*self.args_faker(), **self.kwargs_faker())

class DictProducer(AbstractProducer):
    """
    Produces a hard-coded (inflexible size/keys) dictionary
    """

    def __init__(self, **key_to_faker):
        self.key_to_faker = key_to_faker

    def __call__(self):
        return {
            key : try_or_return(self.key_to_faker[key])
            for key in self.key_to_faker
        }

class ListProducer(AbstractProducer):
    """
    Produces length-invariant list.
    """

    def __init__(self, fakers, weights=None, len=1):
        self.fakers = fakers
        self.weights = weights
        self.len = len

    def __call__(self):
        fakers = rand.choices(self.fakers, self.weights, k=self.len)
        return [try_or_return(x) for x in fakers]

class UniformProducer(AbstractProducer):

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __call__(self):
        return rand.uniform(self.lower, self.upper)

class NormalProducer(AbstractProducer):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return rand.gauss(self.mu, self.sigma)

#endregion

#region PREDEFINED PRODUCERS

StandardUniform = UniformProducer(0, 1)
StandardNormal = NormalProducer(0, 1)

BoolCoinFlip = UniformProducer(0, 1)\
    .then(lambda x : x > 0.5)

#endregion
