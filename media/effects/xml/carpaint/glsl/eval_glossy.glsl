vec3 glossy_eval(vec3 albedo, vec3 wo, vec3 ns, vec3 wi)
{
  vec3 wh = normalize(wo + wi);
  
  float D = shininess * 2.0f * pow(abs(dot(ns, wh)), shininess);

  // compute some dot products
  float n_dot_wo  = abs(dot(ns, wo));
  float n_dot_wi  = abs(dot(ns, wi));
  float wo_dot_wh = abs(dot(wo, wh));
  float one_over_wo_dot_wh = 1.0f / wo_dot_wh;

  // evaluate shadow masking term
  float two_times_n_dot_wh_over_wo_dot_wh = 2.0f * abs(dot(ns, wh)) * one_over_wo_dot_wh;
  float G = min(1.0f, min(n_dot_wo * two_times_n_dot_wh_over_wo_dot_wh,
                          n_dot_wi * two_times_n_dot_wh_over_wo_dot_wh));
  
  // Fresnel is fresnel_no_op here, F == 1.0f factor removed.
  return (albedo * G * D) / (4.0f * n_dot_wo * n_dot_wi);
}