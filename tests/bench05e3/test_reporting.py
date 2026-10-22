"""Preserve acquisition/scope denominators when E3 joins historical results."""
import unittest
from inference_kit.analysis.index_report import _quality_overview

class ReportingScopeTests(unittest.TestCase):
    def test_historical_and_current_same_arm_metric_never_pool(self):
        rows=[]
        for analysis,scope,n in [('old','historical_task',24),('new','current_task',2)]:
            rows.extend({'task_id':'task','metric':'task.task_success','value':False,'scope':scope,
                         'analysis_id':analysis,'coordinates':{'arm_id':'same-arm'}} for _ in range(n))
        html=_quality_overview(rows,lambda row:row['task_id'])
        self.assertIn('historical_task',html);self.assertIn('current_task',html)
        self.assertIn('observed_row_denominator&quot;: 24',html)
        self.assertIn('observed_row_denominator&quot;: 2',html)
        self.assertNotIn('observed_row_denominator&quot;: 26',html)

    def test_analysis_revisions_remain_separate(self):
        rows=[{'task_id':'task','metric':'task.acc','value':1.0,'scope':'current_task','analysis_id':a,
               'coordinates':{'arm_id':'same'}} for a in ('rev1','rev2')]
        html=_quality_overview(rows,lambda row:row['task_id'])
        self.assertIn('rev1',html);self.assertIn('rev2',html)
        self.assertEqual(html.count('observed_row_denominator&quot;: 1'),2)

if __name__=='__main__':unittest.main()
