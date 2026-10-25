from evaluation.reports import build_report, write_reports, aggregate_reports

def test_write_reports(tmp_path):
    report = build_report("a.wav", "ref", [{"model":"m", "text":"pred", "device":"cpu", "detected_language":None, "elapsed_seconds":1, "wer":0, "cer":0}])
    jpath, cpath = write_reports(report, tmp_path)
    assert jpath.exists() and cpath.exists()

def test_aggregate_excludes_unscored_rows():
    reports = [build_report("a", "r", [{"model":"m", "wer":.2, "cer":.1}]), build_report("b", "", [{"model":"m", "wer":None, "cer":None}])]
    assert aggregate_reports(reports) == [{"model":"m", "recordings_run":2, "recordings_scored":1, "mean_wer":.2, "mean_cer":.1}]
