
#ifndef PI
#define PI  3.14159265358979f
#endif

float fresnel_schlick(float cos_theta, float exponent, float minimum, float maximum)
{
  // The max doesn't seem like it should be necessary, but without it you get
  // annoying broken pixels at the center of reflective spheres where cos_theta ~ 1.
  return clamp(minimum + (maximum - minimum) * pow(max(0.0f, 1.0f - cos_theta), exponent), minimum, maximum);
}

float intensity(vec3 rgb)
{
  return (rgb.x + rgb.y + rgb.z) * 0.333333f;
}


void main(void)
{
  vec3 pos = varWorldPos;                 // surface hit point
  vec3 wo  = normalize(varEyePos - pos);  // vector from surface point to observer
  vec3 ns  = normalize(varNormal);        // shading normal

  float wo_dot_ns = max(0.0f, dot(wo, ns));
  vec3 diffuse_albedo = mix(diffuseFade.xyz, diffuse.xyz, wo_dot_ns); 

  float fresnel = fresnel_schlick(wo_dot_ns, 1.0f + reflectivityFalloff * 4.0f, reflectivityMin, reflectivityMax);
  
  vec3 rgb = vec3(0.0f);
    
  // Unsized array, only as many lights in this loop as there are enabled
  if ( gl_FrontFacing )
  {
    float att_i  = 1.0f;  // distance attenuation factor
    float spot_i = 1.0f;  // spotlight attenuation factor
    
    vec3 lightAmbient;
    vec3 lightDiffuse;
    vec3 lightSpecular;
    vec3 wi;
    for (int i = 0; i < sys_NumLights; ++i)
    {
      sampleLight(sys_Lights[i], varWorldPos, wi, lightAmbient, lightDiffuse, lightSpecular);
      
      float ns_dot_wi = max(0.0f, dot(ns, wi));
      vec3 bsdf_val_dot = (att_i * spot_i * ns_dot_wi * (1.0f - fresnel)) *  // multiply the single floats first.
                          (diffuse_albedo * lightDiffuse + glossy_eval(reflective.xyz, wo, ns, wi) * lightSpecular.xyz);
      
      if (0.0f < intensity(bsdf_val_dot))
      {
        rgb += bsdf_val_dot;
      }
    }
    // "Specular Reflection" emulated with a 2D spherical environment map.
    if (sys_EnvironmentSamplerEnabled)
    {
      vec3 r  = reflect( -wo, ns );
      vec2 tc = vec2( ( atan( r.x, -r.z ) + PI ) / ( 2.0f * PI ), acos( -r.y ) / PI );
      rgb += fresnel * texture(sys_EnvironmentSampler, tc).xyz * specular.xyz;
    }
  }

  Color = vec4(rgb, 1.0f);
}
