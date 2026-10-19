import pytest
import yaml
from pydantic import ValidationError

from looklab_api.config import AIPolicy, load_ai_policy
from tests.conftest import REPO


def test_shipped_policy_is_open_source_first_and_zero_budget():
    p = load_ai_policy(REPO / "config" / "ai.yaml")
    assert p.default_policy == "open_source_first"
    assert p.spending.monthly_limit_usd == 0
    assert p.virtual_tryon.paid_fallback == "disabled"
    assert p.virtual_tryon.enabled_paid_providers == []
    assert p.creative_editing.enabled is False


def test_unknown_provider_name_is_rejected(tmp_path):
    f = tmp_path / "ai.yaml"
    f.write_text(yaml.safe_dump({"ai": {"virtual_tryon": {"primary": "local_fashn_v2"}}}))
    with pytest.raises(ValidationError):
        load_ai_policy(f)


def test_approval_cannot_be_disabled():
    with pytest.raises(ValidationError):
        AIPolicy.model_validate({"spending": {"require_explicit_approval": False}})


def test_paid_fallback_must_be_explicitly_enabled():
    with pytest.raises(ValidationError):
        AIPolicy.model_validate({"virtual_tryon": {"paid_fallback": "fashn_api"}})
    ok = AIPolicy.model_validate({"virtual_tryon": {"paid_fallback": "fashn_api", "enabled_paid_providers": ["fashn_api"]}})
    assert ok.virtual_tryon.paid_fallback == "fashn_api"


def test_auto_retry_of_paid_is_not_configurable():
    with pytest.raises(ValidationError):
        AIPolicy.model_validate({"spending": {"auto_retry_paid": True}})
