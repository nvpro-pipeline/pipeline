
#ifndef PI
#define PI  3.14159265358979f
#endif

float intensity(vec3 rgb)
{
  return (rgb.x + rgb.y + rgb.z) * 0.333333f;
}

float fresnel_dielectric(float etai, float etat, float cosi)
{
  // start with sint
  float sint = 1.0f - cosi * cosi;
  sint = (sint < 0.0f) ? 0.0f : sint;
  sint = (etai / etat) * sqrt(sint);

  // handle total internal reflection
  if (1.0f < sint)
  {
    return 1.0f;
  }

  float cost = 1.0f - sint * sint;
  cost = (cost < 0.0f) ? 0.0f : cost;
  cost = sqrt(cost);
  
  float rParallel      = (etat * cosi - etai * cost) / (etat * cosi + etai * cost);
  float rPerpendicular = (etai * cosi - etat * cost) / (etai * cosi + etat * cost);

  float result = (rParallel * rParallel + rPerpendicular * rPerpendicular) / 2.0f;
  result = (1.0f < result) ? 0.5f : result;

  return result;
}

void main(void)
{
  vec3 pos = varWorldPos;                 // surface hit point
  vec3 wo  = normalize(varEyePos - pos);  // vector from surface point to observer
  vec3 ns  = normalize(varNormal);        // shading normal
  
  // INFO requires OIT for correct rendering for objects like spheres.
  if ( !gl_FrontFacing )
  {
    ns = -ns;
  }
  float wo_dot_ns = max(0.0f, dot(wo, ns));
  // No direct lighting. Only reflections and tint color affect the surface.

  float fresnel = fresnel_dielectric(1.0f, IOR, wo_dot_ns);

  // "Specular Reflection" emulated with a 2D spherical environment map.
  vec3 reflectance = vec3(0.0f);
  if (sys_EnvironmentSamplerEnabled)
  {
    vec3 r  = reflect(-wo, ns);
    vec2 tc = vec2( ( atan( r.x, -r.z ) + PI ) / ( 2.0f * PI ), acos( -r.y ) / PI );
    reflectance = texture(sys_EnvironmentSampler, tc).xyz * reflectiveColor.xyz;
  }

  vec3 result = mix(transparentColor.xyz, reflectance, fresnel);

  float alpha = min(fresnel + 1.0f - intensity(transparentColor.xyz), 1.0f);
  emitColor( vec4(result,alpha) );
}
