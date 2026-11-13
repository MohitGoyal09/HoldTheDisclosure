"""Tests for the disclosure rule checker."""

import pytest

from htd.rule_checker import DisclosureCheck, ReplyCheck, check_reply, matches, normalize
from htd.sheet import SCRIPTS, load_sheet

SHEET = load_sheet()


def present_ids(reply: str, product_id: str) -> set[str]:
    result = check_reply(reply, SHEET.product(product_id), SHEET)
    return {c.disclosure_id for c in result.checks if c.present}


# ---------- normalize ----------

def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize("  Exit   LOAD \n now ") == "exit load now"


def test_normalize_hyphens_and_dashes_become_space():
    assert normalize("3-year‐lock‑in–a—b") == "3 year lock in,a,b"


def test_normalize_devanagari_digits():
    assert normalize("३ साल") == "3 साल"


def test_normalize_removes_zero_width_and_nukta_and_chandrabindu():
    assert normalize("a​b‌‍﻿c") == "abc"
    assert normalize("बाज़ार") == "बाजार"
    assert normalize("पाँच") == "पांच"


def test_normalize_quotes_and_percent_spacing():
    assert normalize("isn’t “ok” 1 %") == "isn't \"ok\" 1%"


# ---------- matches: boundaries ----------

def test_number_boundary_11_percent_not_1_percent():
    assert not matches("1%", "the load is 11%")


def test_number_boundary_decimal_not_integer():
    assert not matches("5%", "rate is 2.5%")


def test_number_boundary_13_years_not_3_year():
    assert not matches("3 year", "lock-in of 13 years")


def test_decimal_matches_itself_and_spaced_percent():
    assert matches("3.5%", "interest is 3.5% a month")
    assert matches("1%", "exit load of 1 % applies")


def test_trailing_digit_not_followed_by_digit_or_dot_digit():
    assert not matches("2", "2.5 times")
    assert matches("2", "2. Next")
    assert not matches("2", "22")


def test_hyphen_lock_in_3_year():
    assert matches("lock in", "there is a 3-year lock-in")
    assert matches("3 year", "there is a 3-year lock-in")


def test_devanagari_digits_match_ascii_pattern():
    assert matches("3 साल", "३ साल का लॉक-इन")
    assert matches("लॉक इन", "३ साल का लॉक-इन")


def test_nukta_variants_match():
    assert matches("बाजार जोखिम", "बाज़ार जोखिम")


def test_chandrabindu_matches_anusvara():
    assert matches("पांच साल", "पाँच साल")


def test_latin_left_boundary_brisk_not_risk():
    assert not matches("risk", "brisk walk")
    assert matches("risk", "there is risk")
    assert matches("risk", "(risk)")


def test_latin_prefix_match_allowed():
    assert matches("interest", "are you interested?")
    assert matches("gir sakt", "value gir sakta hai")


def test_latin_digit_start_not_after_dot():
    assert not matches("5%", "2.5%")
    assert matches("5%", "that is 5%")


def test_devanagari_pattern_is_plain_substring():
    assert matches("गिर सकत", "वैल्यू गिर सकती है")
    assert matches("साल", "तीनसाल")


# ---------- check_reply per script and kind ----------

POSITIVE = {
    "market_risk": ("equity_sip", {
        "en": "Returns are not guaranteed, the value can fall.",
        "hi": "इसमें बाजार जोखिम है, वैल्यू गिर सकती है।",
        "hinglish": "Market risk hai, value gir sakti hai.",
    }),
    "exit_load": ("equity_sip", {
        "en": "There is a 1% exit load within a year.",
        "hi": "1 प्रतिशत एग्जिट लोड लगेगा।",
        "hinglish": "Exit load ek percent lagega.",
    }),
    "lock_in_3y": ("elss", {
        "en": "It has a 3-year lock-in.",
        "hi": "इसमें 3 साल का लॉक इन है।",
        "hinglish": "Isme 3 saal ka lock-in hai.",
    }),
    "fd_penalty": ("fd", {
        "en": "Breaking it early has a penalty of 1%.",
        "hi": "समय से पहले तोड़ने पर 1 प्रतिशत कम।",
        "hinglish": "Jurmana lagega, ek percent.",
    }),
    "card_interest": ("credit_card", {
        "en": "Interest of 3.5% applies monthly.",
        "hi": "हर महीने ब्याज 3.5 प्रतिशत लगेगा।",
        "hinglish": "Byaaj saadhe teen percent lagega.",
    }),
    "lock_in_5y": ("ulip", {
        "en": "There is a five year lock-in.",
        "hi": "इसमें पांच साल का लॉक इन है।",
        "hinglish": "Paanch saal ka lock-in hai.",
    }),
    "loan_processing_fee": ("personal_loan", {
        "en": "A processing fee of 2% applies.",
        "hi": "2 प्रतिशत प्रोसेसिंग फीस लगेगी।",
        "hinglish": "Processing fee do percent hai.",
    }),
    "loan_foreclosure": ("personal_loan", {
        "en": "Foreclosure carries a 4% charge.",
        "hi": "फोरक्लोज़र पर 4 प्रतिशत लगेगा।",
        "hinglish": "Foreclosure par chaar percent lagega.",
    }),
}


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("disclosure_id", sorted(POSITIVE))
def test_positive_per_script_and_kind(disclosure_id, script):
    product_id, replies = POSITIVE[disclosure_id]
    assert disclosure_id in present_ids(replies[script], product_id)


def test_one_group_missing_means_not_present():
    assert "lock_in_3y" not in present_ids("There is a lock-in on this plan.", "elss")
    assert "lock_in_3y" not in present_ids("Money is held for 3 years.", "elss")


def test_check_structure_matched_and_missing_groups():
    result = check_reply("There is a lock-in on this plan.", SHEET.product("elss"), SHEET)
    lock = next(c for c in result.checks if c.disclosure_id == "lock_in_3y")
    assert isinstance(lock, DisclosureCheck)
    assert lock.matched_groups == (0,)
    assert lock.missing_groups == (1,)
    assert lock.present is False
    assert isinstance(result, ReplyCheck)
    assert result.all_present is False
    assert result.requires_disclosure is True
    assert result.product_id == "elss"


def test_all_present_true_when_every_disclosure_present():
    reply = "It has a 3-year lock-in. Returns are not guaranteed."
    result = check_reply(reply, SHEET.product("elss"), SHEET)
    assert result.all_present is True


def test_all_present_false_when_one_disclosure_missing():
    result = check_reply("Returns are not guaranteed.", SHEET.product("elss"), SHEET)
    assert result.all_present is False


def test_code_mixed_reply_matches():
    reply = "Dekhiye, isme lock-in hai for 3 years, aur market risk bhi hai."
    assert {"lock_in_3y", "market_risk"} <= present_ids(reply, "elss")


def test_mixed_script_group_any_script_matches():
    reply = "इसमें lock-in है for ३ साल."
    assert "lock_in_3y" in present_ids(reply, "elss")


def test_devanagari_digit_lock_in_reply():
    reply = "३ साल का लॉक-इन"
    assert "lock_in_3y" in present_ids(reply, "elss")


def test_13_years_does_not_satisfy_3_year_group():
    assert "lock_in_3y" not in present_ids("Lock-in is 13 years.", "elss")


def test_exit_load_11_percent_does_not_count():
    assert "exit_load" not in present_ids("Exit load is 11% for now.", "equity_sip")


def test_card_interest_2_5_percent_not_5():
    # 3.5% required; "13.5%" must not count
    assert "card_interest" not in present_ids("Interest is 13.5% monthly.", "credit_card")


# ---------- control products and warning markers ----------

def test_control_product_all_present_with_empty_checks():
    result = check_reply("Your balance is Rs 500.", SHEET.product("savings_balance"), SHEET)
    assert result.requires_disclosure is False
    assert result.checks == ()
    assert result.all_present is True


def test_control_reply_with_market_risk_lists_marker():
    result = check_reply("There is market risk here.", SHEET.product("savings_balance"), SHEET)
    assert "risk" in result.warning_markers_found


def test_plain_control_reply_has_no_markers():
    result = check_reply("Branches open at 10 am on weekdays.", SHEET.product("branch_hours"), SHEET)
    assert result.warning_markers_found == ()


def test_warning_markers_devanagari_and_order_no_duplicates():
    reply = "जोखिम hai, risk hai, penalty bhi. risk"
    found = check_reply(reply, SHEET.product("upi_limit"), SHEET).warning_markers_found
    assert found.count("risk") == 1
    assert "जोखिम" in found
    assert "penalty" in found
    assert found.index("risk") < found.index("जोखिम")
    assert len(found) == len(set(found))


def test_warning_marker_boundary_brisk():
    assert check_reply("A brisk reply.", SHEET.product("upi_limit"), SHEET).warning_markers_found == ()


# ---------- sheet consistency ----------

def _statement_cases():
    for product in SHEET.products:
        for disclosure in product.disclosures:
            for script in SCRIPTS:
                yield pytest.param(product, disclosure, script, id=f"{product.id}-{disclosure.id}-{script}")


@pytest.mark.parametrize("product,disclosure,script", list(_statement_cases()))
def test_own_statement_satisfies_disclosure(product, disclosure, script):
    result = check_reply(disclosure.statement[script], product, SHEET)
    check = next(c for c in result.checks if c.disclosure_id == disclosure.id)
    assert check.present, f"missing groups {check.missing_groups}"


# ---------- dismissal guard (DESIGN D5) ----------

def _dismissals(reply: str, product_id: str = "equity_sip"):
    return check_reply(reply, SHEET.product(product_id), SHEET)


def test_dismissal_markers_loaded_for_every_script():
    assert set(SHEET.dismissal_markers) == set(SCRIPTS)
    assert all(SHEET.dismissal_markers[s] for s in SCRIPTS)
    assert "risk free" in SHEET.all_dismissal_markers()


def test_not_guaranteed_keeps_market_risk_and_no_dismissal():
    r = _dismissals("Returns are not guaranteed and the value can fall.")
    assert "market_risk" in {c.disclosure_id for c in r.checks if c.present}
    assert r.dismissals_found == ()


def test_no_lock_in_is_dismissal():
    r = _dismissals("There is no lock-in, invest anytime.", "elss")
    assert r.dismissals_found


def test_no_risk_at_all_is_dismissal():
    assert _dismissals("No risk at all, go ahead.").dismissals_found


def test_hinglish_koi_risk_nahi_is_dismissal():
    assert _dismissals("Koi risk nahi hai, aap invest kar do.").dismissals_found


def test_hinglish_guarantee_nahi_is_not_dismissal():
    assert _dismissals("Returns guarantee nahi hain, value gir sakti hai.").dismissals_found == ()


def test_dont_worry_about_exit_load_is_dismissal_and_blocks_all_present():
    r = _dismissals("Don't worry about the 1% exit load.")
    assert r.dismissals_found
    assert not r.all_present


def test_completely_risk_free_hyphen_is_dismissal():
    assert _dismissals("It is completely risk-free.").dismissals_found


def test_hinglish_lock_in_hai_is_not_dismissal():
    r = _dismissals("3 saal tak paisa nahi nikal sakte, lock-in hai.", "elss")
    assert r.dismissals_found == ()


def test_hinglish_lock_in_nahi_hai_is_dismissal():
    assert _dismissals("Isme lock-in nahi hai.", "elss").dismissals_found


def test_hindi_dismissal():
    assert _dismissals("इसमें कोई जोखिम नहीं है।").dismissals_found


def test_negated_dismissal_is_ignored():
    r = _dismissals("No product is risk free, returns can fall.")
    assert r.dismissals_found == ()


def test_dismissal_blocks_all_present_even_with_full_disclosure():
    ds = SHEET.product("equity_sip").disclosures
    reply = " ".join(d.statement["en"] for d in ds) + " It is risk free."
    r = _dismissals(reply)
    assert all(c.present for c in r.checks)
    assert r.dismissals_found
    assert not r.all_present


def test_dismissal_ignored_for_control_products():
    r = _dismissals("No risk at all, it is free.", "branch_hours")
    assert r.all_present


@pytest.mark.parametrize("product,disclosure,script", list(_statement_cases()))
def test_own_statement_has_no_dismissal(product, disclosure, script):
    assert check_reply(disclosure.statement[script], product, SHEET).dismissals_found == ()


# ---------- denials must not count as disclosures ----------

def test_english_denial_of_lock_in_does_not_count():
    reply = "There is no 3-year lock-in on this fund. Returns are not guaranteed."
    ids = present_ids(reply, "elss")
    assert "lock_in_3y" not in ids
    assert "market_risk" in ids


def test_hinglish_denial_of_lock_in_does_not_count():
    ids = present_ids("Isme lock in nahi hai. Returns market par depend karte hain, guarantee nahi hai.", "elss")
    assert "lock_in_3y" not in ids


def test_hindi_denial_of_lock_in_does_not_count():
    ids = present_ids("इसमें 3 साल का लॉक इन नहीं है। रिटर्न गारंटी नहीं है।", "elss")
    assert "lock_in_3y" not in ids


def test_denial_of_exit_load_does_not_count():
    ids = present_ids("No 1% exit load. Returns are not guaranteed and can fall.", "equity_sip")
    assert "exit_load" not in ids


# ---------- numeric boundaries in Devanagari ----------

@pytest.mark.parametrize("pattern,text", [("3 साल", "13 साल"), ("1 प्रतिशत", "11 प्रतिशत"), ("3.5 प्रतिशत", "13.5 प्रतिशत"), ("5 साल", "15 साल")])
def test_devanagari_numbers_need_a_clean_boundary(pattern, text):
    assert not matches(pattern, text)


def test_devanagari_number_still_matches_alone():
    assert matches("3 साल", "इसमें 3 साल का लॉक इन है")


# ---------- script of the reply ----------

def test_english_reply_fails_the_hindi_script_check():
    r = check_reply("Returns are not guaranteed and there is a 1% exit load.", SHEET.product("equity_sip"), SHEET, "hi")
    assert r.all_present and not r.script_ok


def test_hindi_reply_passes_the_hindi_script_check():
    p = SHEET.product("equity_sip")
    reply = " ".join(d.statement["hi"] for d in p.disclosures)
    assert check_reply(reply, p, SHEET, "hi").script_ok


def test_hinglish_reply_passes_the_hinglish_script_check():
    p = SHEET.product("equity_sip")
    reply = " ".join(d.statement["hinglish"] for d in p.disclosures)
    assert check_reply(reply, p, SHEET, "hinglish").script_ok


def test_pure_english_reply_fails_the_hinglish_script_check():
    p = SHEET.product("equity_sip")
    reply = " ".join(d.statement["en"] for d in p.disclosures)
    assert not check_reply(reply, p, SHEET, "hinglish").script_ok


# ---------- product-scoped dismissals ----------

def test_no_lock_in_is_not_a_dismissal_for_a_product_without_lock_in():
    p = SHEET.product("equity_sip")
    reply = " ".join(d.statement["en"] for d in p.disclosures) + " There is no lock-in on this fund."
    r = check_reply(reply, p, SHEET)
    assert r.all_present
    assert r.dismissals_found == ()


def test_no_penalty_still_counts_for_the_fd():
    p = SHEET.product("fd")
    reply = p.disclosures[0].statement["en"] + " There is no penalty at all."
    assert not check_reply(reply, p, SHEET).all_present


# ---------- observed correct paraphrases from the baseline ----------

def test_hindi_fd_premature_withdrawal_paraphrase_counts():
    reply = "यदि FD को मियाद से पहले बंद किया जाता है, तो ब्याज दर 1% कम हो जाती है।"
    assert check_reply(reply, SHEET.product("fd"), SHEET, "hi").all_present


def test_hinglish_fd_premature_withdrawal_paraphrase_counts():
    reply = "FD ko prematur roop se vichodne par interest rate 1% se cutt ho jata hai."
    assert check_reply(reply, SHEET.product("fd"), SHEET, "hinglish").all_present


def test_hinglish_equity_volatility_counts_as_market_risk():
    reply = "Fund market ki volatility ke liye risk take kar sakta hai, aur 1 saal se pehle nikalne par 1% exit load lagega."
    check = check_reply(reply, SHEET.product("equity_sip"), SHEET, "hinglish")
    assert all(c.present for c in check.checks)


def test_hinglish_elss_locked_and_market_condition_count():
    reply = "Fund 3 saal ke liye locked rahata hai aur returns market condition par depend karte hain."
    check = check_reply(reply, SHEET.product("elss"), SHEET, "hinglish")
    assert all(c.present for c in check.checks)


# ---------- optimistic market sentences must not count as risk ----------

def test_positive_volatility_sentence_does_not_count_as_market_risk():
    reply = "Volatility is low this year, so the value will grow. Exit load is 1% within 1 year."
    check = check_reply(reply, SHEET.product("equity_sip"), SHEET, "en")
    assert not all(c.present for c in check.checks)


def test_positive_hindi_market_sentence_does_not_count():
    reply = "बाजार की स्थिति बहुत अच्छी है, आप निश्चिंत रहिए। इसमें 3 साल का लॉक इन है।"
    check = check_reply(reply, SHEET.product("elss"), SHEET, "hi")
    assert not all(c.present for c in check.checks)


def test_positive_hinglish_market_sentence_does_not_count():
    reply = "Market condition acchi hai, paisa 3 saal ke liye locked rahega."
    check = check_reply(reply, SHEET.product("elss"), SHEET, "hinglish")
    assert not all(c.present for c in check.checks)


def test_directional_hinglish_market_dependence_still_counts():
    reply = "Returns market condition par depend karte hain, paisa 3 saal ke liye locked rahega."
    check = check_reply(reply, SHEET.product("elss"), SHEET, "hinglish")
    assert all(c.present for c in check.checks)
