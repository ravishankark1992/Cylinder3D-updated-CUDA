"""gt=bwlabel(GT); pm=bwlabel(PM); gt=gt(:); pm=pm(:);
n_gt=max(gt); n_pm=max(pm); max_instances=max(n_gt,n_pm)+1;
% span ids up to the max to handle non-contiguous IDs; missing will have
%  area 0
[gt_areas,~]=hist(gt,0:n_gt); [pm_areas, ~]=hist(pm,0:n_pm);
% define an unique code for each PM/GT pair, unique values are intersections
intrs=gt.*max_instances+pm; [intr_areas,intr_ids]=hist(intrs,unique(intrs));

TP=0; iou=0;
for i=1:length(intr_ids)
  intr_id=intr_ids(i);
  gt_id=fix(intr_id/max_instances)+1;
  pm_id=rem(intr_id, max_instances)+1;
  if gt_id==1||pm_id==1, continue; end  % skip background
  intr=intr_areas(i);
  % remove intersection with background (first max_instance IDs)
  unio=gt_areas(gt_id)+pm_areas(pm_id)-intr;
  if unio>0, iou_i=intr/unio; else, iou_i=0; end
  if iou_i>0.5, iou=iou+iou_i; TP=TP+1; end  % match only for IoU>0.5
end

% all GTs/predictions: non-null areas except background
FN=sum(gt_areas>0)-TP-1; FP=sum(pm_areas>0)-TP-1;
if TP>0, sq=iou/TP; else, sq=0; end
rq=TP/(TP+0.5*FP+0.5*FN);
pq=sq*rq;

"""