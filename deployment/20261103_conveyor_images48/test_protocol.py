import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import identity
import prepare
from protocol import (ALLOWED_GPU_PRODUCTS, CONTROLS, JOB_NAME,NAMESPACE, episode_dir,
                      SAM3_COMMIT, assignment_identity,control_plan,source_files,verify_freeze)


class ProtocolTests(unittest.TestCase):
    def test_dynamic_mapping_allows_non_modulo_lane_and_control_is_hardware_fixed(self):
        mapping={'layouts':{str(i):{'layout':i,'gpu':(i+3)%8,'gpu_uuid':'GPU-'+str((i+3)%8),
            'pod_uid':'pod','node':'node','treatment_task_wall_s':i} for i in range(55)}}
        plans=[x for gpu in range(8) for x in control_plan(gpu,mapping)]
        self.assertEqual(CONTROLS['stage_order'],['conditions','control'])
        self.assertEqual(sorted(x['layout'] for x in plans),list(range(55)))
        entry=assignment_identity(0,3,mapping,{'uid':'pod','node':'node'},'GPU-3')
        self.assertEqual(entry['gpu'],3)
        self.assertEqual([x['layout'] for x in control_plan(3,mapping)],sorted([i for i in range(55) if (i+3)%8==3],reverse=True))
        with self.assertRaises(RuntimeError):assignment_identity(0,0,mapping)
        with self.assertRaises(RuntimeError):assignment_identity(0,3,mapping,actual_uuid='GPU-other')

    def test_bad_identity_arguments_are_rejected(self):
        for args in ((-1,0),(55,0),(0,8),(True,0)):
            with self.assertRaises(ValueError): assignment_identity(*args,{})
        with self.assertRaises(RuntimeError):control_plan(0,{'layouts':{}})
        with self.assertRaises(ValueError): episode_dir('full','unknown',0)

    def test_shared_queue_has_no_fixed_node_or_priority_override(self):
        docs=prepare.manifests()
        self.assertEqual([d['kind'] for d in docs],['ServiceAccount','Role','RoleBinding','Job'])
        self.assertTrue(prepare.validate_job(docs[-1]))
        spec=docs[-1]['spec']['template']['spec']
        self.assertNotIn('nodeName',spec)
        self.assertNotIn('nodeSelector',spec)
        self.assertNotIn('tolerations',spec)
        self.assertNotIn('priorityClassName',spec)
        self.assertNotIn('preemptionPolicy',spec)
        self.assertEqual(spec['containers'][0]['resources']['limits']['nvidia.com/gpu'],'8')
        self.assertEqual(docs[1]['rules'],[{'apiGroups':[''],'resources':['pods'],'verbs':['get']}])
        self.assertNotIn('api_key',json.dumps(docs))
        unsupported=copy.deepcopy(docs[-1])
        unsupported['spec']['template']['spec']['preemptionPolicy']='Never'
        with self.assertRaises(AssertionError):prepare.validate_job(unsupported)

    def fixture(self):
        job=prepare.manifests()[-1]
        pod={'metadata':{'name':'example','uid':'pod-uid','namespace':NAMESPACE},
             'spec':copy.deepcopy(job['spec']['template']['spec']), 'status':{'phase':'Running'}}
        pod['metadata']['ownerReferences']=[{'kind':'Job','controller':True,'name':JOB_NAME,'uid':'job-uid'}]
        pod['spec']['nodeName']='192.168.48.200'
        pod['spec']['preemptionPolicy']='PreemptLowerPriority'  # cluster admission default
        pod['spec']['tolerations']=[{'key':'node.kubernetes.io/not-ready','effect':'NoExecute','operator':'Exists'}]
        env={'HARNESS_POD_NAME':'example','HARNESS_POD_UID':'pod-uid',
             'HARNESS_POD_NAMESPACE':NAMESPACE,'HARNESS_NODE_NAME':'192.168.48.200',
             'HOSTNAME':'example','HARNESS_ALLOWED_GPU_PRODUCTS':','.join(ALLOWED_GPU_PRODUCTS)}
        return pod,env

    def test_actual_pod_node_uid_allocation_and_policy_checked(self):
        pod,env=self.fixture()
        proof=identity.validate_pod(pod,env)
        self.assertEqual(proof['node'],'192.168.48.200')
        for section,key,value in (('metadata','uid','other'),('spec','nodeName','other'),
                                  ('status','phase','Pending')):
            wrong=copy.deepcopy(pod);wrong[section][key]=value
            with self.assertRaises(RuntimeError):identity.validate_pod(wrong,env)
        wrong=copy.deepcopy(pod);wrong['spec']['containers'][0]['resources']['limits']['nvidia.com/gpu']='6'
        with self.assertRaises(RuntimeError):identity.validate_pod(wrong,env)
        wrong=copy.deepcopy(pod);wrong['spec']['priority']=100
        with self.assertRaises(RuntimeError):identity.validate_pod(wrong,env)
        wrong=copy.deepcopy(pod);wrong['spec']['priorityClassName']='custom-priority'
        with self.assertRaises(RuntimeError):identity.validate_pod(wrong,env)

    def test_gpu_inventory_rejects_wrong_count_duplicate_and_hardware(self):
        text='\n'.join(f'{i}, GPU-{i}, 0000:{i:02x}:00.0, NVIDIA GeForce RTX 5090, 30000, 0' for i in range(8))
        with patch.object(identity.subprocess,'check_output',return_value=text):
            self.assertEqual(len(identity.gpu_inventory()),8)
        for wrong in (text.rsplit('\n',1)[0],text.replace('GPU-7','GPU-0'),text.replace('RTX 5090','RTX 4090')):
            with patch.object(identity.subprocess,'check_output',return_value=wrong):
                with self.assertRaises(RuntimeError):identity.gpu_inventory()

    def test_freeze_detects_mutation_and_added_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'policy.py').write_text('value=1\n')
            frozen={'controls':CONTROLS,'allowed_gpu_products':list(ALLOWED_GPU_PRODUCTS),
                    'sources':{'test':{'root':str(root),'files':source_files(root)}},'files':[]}
            verify_freeze(frozen)
            (root/'extra.py').write_text('value=2\n')
            with self.assertRaises(RuntimeError):verify_freeze(frozen)
            (root/'extra.py').unlink()
            (root/'policy.py').write_text('value=3\n')
            with self.assertRaises(RuntimeError):verify_freeze(frozen)

    def test_sam3_editable_registration_uses_versioned_path_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            pyr=Path(temp)/'PyRUA-Lean'
            root=pyr/'runtime/vendor'/('sam3-'+SAM3_COMMIT)
            (root/'sam3').mkdir(parents=True)
            (root/'sam3/__init__.py').write_text('raise RuntimeError("must not import during preparation")\n')
            site=pyr/'runtime/policy-venv/lib/python3.11/site-packages'
            metadata=site/'sam3-0.1.0.dist-info/direct_url.json'
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps({'dir_info':{'editable':True},'url':root.as_uri()}))
            pth=site/'__editable__.sam3-0.1.0.pth'
            pth.write_text('import __editable___sam3_0_1_0_finder; __editable___sam3_0_1_0_finder.install()')
            finder=site/'__editable___sam3_0_1_0_finder.py'
            finder.write_text('MAPPING: dict[str, str] = '+repr({'sam3':str(root/'sam3')})+'\nraise RuntimeError("do not execute")\n')
            registration=prepare.sam3_registration(pyr)
            self.assertEqual(registration['root'],root)
            self.assertEqual(registration['metadata_files'],[metadata,pth,finder])
            finder.write_text('MAPPING: dict[str, str] = {"sam3":"/wrong/package"}\n')
            with self.assertRaises(RuntimeError):prepare.sam3_registration(pyr)


if __name__=='__main__': unittest.main()
