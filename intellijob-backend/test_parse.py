import sys
sys.path.insert(0, 'E:/Glasgow MSC IT Slides/Final sem/intellijob/intellijob-backend')

from api.services.roadmap_generator import _coerce_phase_plan

test_output = '''{
  "phase_name": "Entry-Level Readiness",
  "phase_focus": "Build foundational skills",
  "total_weeks": 10,
  "weeks": [
    {"week": 1, "theme": "Test", "milestones": ["m1"], "skills": ["Python"], "project": "p1"}
  ],
  "key_projects": [
    {"title": "Test Project", "description": "Test desc"}
  ],
  "common_pitfalls": ["pitfall1"],
  "resource_priorities": ["skill1"]
}'''

try:
    result = _coerce_phase_plan(test_output)
    print('Parsed successfully!')
    print('Total weeks:', result.total_weeks)
    print('Key projects:', result.key_projects)
    print('Weeks:', len(result.weeks))
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()