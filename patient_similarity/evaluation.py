def evaluate(patients, rankings, k=5):
    per={}
    for p in patients:
        uid=p["patient_uid"]; rel=p["similar_patients"]
        item={"patient_id":p["patient_id"]}
        for method, result in rankings.items():
            got=result.get(uid,[])[:k]
            hits=len(set(got)&rel)
            item[method]={"precision_5":round(hits/k,6),
                          "recall_5":round(hits/len(rel),6) if rel else None}
        per[uid]=item
    summary={"metrics":{}}
    for method in rankings:
        ps=[v[method]["precision_5"] for v in per.values()]
        rs=[v[method]["recall_5"] for v in per.values()
            if v[method]["recall_5"] is not None]
        summary["metrics"][method]={
            "average_precision_5":round(sum(ps)/len(ps),6) if ps else 0,
            "average_recall_5":round(sum(rs)/len(rs),6) if rs else None,
            "precision_patients_count":len(ps),"recall_patients_count":len(rs)}
    summary["patients_with_no_ground_truth_excluded_from_average_recall"]=sum(
        not p["similar_patients"] for p in patients)
    summary["total_patients"]=len(patients)
    a,b=rankings["tfidf"],rankings["bm25"]
    overlaps=[]
    for uid in a:
        x,y=set(a[uid]),set(b[uid]); overlaps.append(len(x&y)/len(x|y) if x|y else 0)
    summary["average_top5_jaccard_overlap_tfidf_bm25"]=round(sum(overlaps)/len(overlaps),6) if overlaps else None
    return per,summary
