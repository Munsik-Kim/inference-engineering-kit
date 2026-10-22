"""Typed dispatch only; render/score and model lifecycle remain separate."""
from ..core.io import require

async def execute(request, *, session=None,url=None,natural_collector=None,timeout=30,event_sink=None):
    request.validate()
    if request.request_kind=='choice_loglikelihood':
        require(session is not None and url,'Likelihood requires HTTP session and loopback server')
        from .likelihood import collect
        return await collect(session,url,request,timeout,event_sink)
    require(natural_collector is not None,'Generation requires explicit existing natural collector')
    return await natural_collector(request)
