"""Audit regression tests, added for Gravity gold-standard review (2026-10-08)."""
import asyncio
import json
import sys
import types
import pytest
from pydantic import ValidationError
from relayg.graph import build_graph
from relayg.nodes import ground_amount, mock_classify
from relayg.state import Classification
from relayg.tools import payload, ToolRefused
from test_graph import FakeTool, make_tools, run, kw_classifier

@pytest.mark.parametrize('body', ['Refund $1,200 please', 'Refund $12.345 please', 'Refund -$10 please', 'Refund $-10 please'])
def test_bad_money_not_partially_parsed(body):
    assert mock_classify('Refund', body).refund_amount is None

def test_multiple_amounts_manual_review():
    r = run('Refund', 'Refund $20 or $80.', make_tools())
    assert r['refund_amount'] is None
    assert r['policy_decision'] == 'no_refund'
    assert any('ambiguous' in n for n in r['notes'])

def test_charge_question_is_not_refund():
    assert mock_classify('Question', 'Why was I charged $20?').intent == 'question'

@pytest.mark.parametrize('amount', [-1, float('nan'), float('inf')])
def test_invalid_model_money_rejected(amount):
    with pytest.raises(ValidationError):
        Classification(intent='refund_request', urgency='low', refund_amount=amount)

def test_draft_does_not_claim_money_moved():
    t = make_tools(); run('Refund', 'Refund $20 please.', t)
    body=t['gmail_draft'].calls[0]['body']
    assert 'No refund has been issued' in body and 'being processed' not in body

def test_send_approval_not_refund_approval():
    t=make_tools(); r=run('Refund', 'Refund $80 please.', t)
    assert 'No refund has been issued' in t['gmail_send'].calls[0]['body']
    assert r['actions'][0]['status']=='review_needed'
    assert 'approved_by' not in r['actions'][0]

def test_skipped_escalation_honest_draft():
    t=make_tools(); r=run('Refund', 'Refund $800 please.', t)
    assert 'no escalation was delivered' in t['gmail_draft'].calls[0]['body']
    assert 'escalate' in r['unavailable']

def test_refused_escalation_honest_draft():
    t=make_tools(slack_reply='APPROVAL_REJECTED'); run('Refund', 'Refund $800 please.', t, channel='C1')
    assert 'no escalation was delivered' in t['gmail_draft'].calls[0]['body']

class BrokenTool:
    async def ainvoke(self, args):
        raise RuntimeError('network error')

@pytest.mark.parametrize('key,body,channel', [('gmail_send','Refund $80',''),('slack_send','Refund $800','C1'),('gmail_draft','Refund $20','')])
def test_tool_exception_reported(key,body,channel):
    t=make_tools();t[key]=BrokenTool();r=run('Refund',body,t,channel=channel)
    assert any(a['status']=='failed' for a in r['actions'])
    assert r['unavailable']
    if key=='gmail_send': assert t['gmail_draft'].calls

def test_missing_tool_reported():
    t=make_tools();del t['gmail_send'];r=run('Refund','Refund $80',t)
    assert any('KeyError' in n for n in r['notes'])
    assert t['gmail_draft'].calls

@pytest.mark.parametrize('raw',[None,[],{},'null','[]',{'isError':True},{'successful':False,'error':'fail'}])
def test_invalid_tool_result_fails_closed(raw):
    with pytest.raises(ToolRefused): payload(raw)

def test_input_limit_no_model_or_tool_calls():
    async def forbidden(*args): raise AssertionError('should not classify')
    t=make_tools();r=run('Refund','x'*20001,t,classifier=forbidden)
    assert r['classifier']=='not-run' and r['limits']['tickets_processed']==0
    assert all(not v.calls for v in t.values())

def test_result_sources_and_summary():
    r=run('Refund','Refund $20',make_tools())
    assert r['summary'].startswith('Ticket T-1:')
    assert r['claim_sources']['refund_amount']=='ticket'
    assert r['sources'][0]['ticket_id']=='T-1'

def test_second_invocation_does_not_reuse_actions():
    g=build_graph(kw_classifier,make_tools())
    async def both():
        a=await g.ainvoke({'inputs':{'ticket_id':'one','subject':'Refund','body':'Refund $80','customer_email':'c@example.com'}})
        a['inputs']={'ticket_id':'two','subject':'Help','body':'How?','customer_email':'c@example.com'}
        return await g.ainvoke(a)
    r=asyncio.run(both())['result']
    assert r['approved'] is False
    assert len(r['actions'])==1

class Pause(BaseException): pass
class PausingTool:
    async def ainvoke(self,args): raise Pause()
def test_hosted_control_signal_not_swallowed():
    t=make_tools();t['gmail_send']=PausingTool()
    with pytest.raises(Pause): run('Refund','Refund $80',t)

def test_graph_uses_hosted_sdk_when_available(monkeypatch):
    import agent
    class Model:
        def __init__(self,**kwargs): pass
        def with_structured_output(self,*args): return self
    called=[]
    async def hosted_tools(): called.append('hosted'); return []
    monkeypatch.setitem(sys.modules,'gravity',types.SimpleNamespace(tools=hosted_tools))
    monkeypatch.setattr(agent,'ChatOpenAI',Model)
    for key in ['GRAVITY_RUN_TOKEN','GRAVITY_LLM_URL','GRAVITY_GATEWAY_URL']:
        monkeypatch.setenv(key,'http://example.invalid')
    assert asyncio.run(agent.graph()) is not None
    assert called==['hosted']


def test_invented_refund_intent_not_accepted():
    async def lying(subject,body):
        return Classification(intent='refund_request',urgency='low',refund_amount=20)
    r=run('Question','Why was I charged $20?',make_tools(),classifier=lying)
    assert r['policy_decision']=='no_refund' and r['refund_amount'] is None

def test_unknown_amount_reason_honest():
    r=run('Refund','Please refund me.',make_tools())
    assert 'missing' in r['policy_reason']
