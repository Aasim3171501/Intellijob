"""
IntelliJob — URL routing for the api app.

Single endpoint in Phase B-2:

    POST /api/analyze/   ->  extract skills + match jobs + roadmap

Mounted from core/urls.py via include('api.urls').
"""

from __future__ import annotations

from django.urls import path

from api.views import AnalyzeView, LearningResourcesView, PhasePlanView

app_name = "api"

urlpatterns = [
    path("analyze/", AnalyzeView.as_view(), name="analyze"),
    path("learning-resources/", LearningResourcesView.as_view(), name="learning-resources"),
    path("phase-plan/", PhasePlanView.as_view(), name="phase-plan"),
]