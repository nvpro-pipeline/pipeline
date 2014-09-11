// Phong lighting
vec3 eval(in vec3 wo, in vec3 ns, in vec3 wi, 
          in vec3 ambientColor, in vec3 diffuseColor, in vec3 specularColor,
          in vec3 ambient, in vec3 diffuse, in vec3 specular, in float exponent)
{
  float shine = 0.0;
  float ns_dot_wi = max(0.0, dot(ns, wi));
  if (0.0f < ns_dot_wi)
  {
    // Phong
    vec3 R = reflect(-wi, ns);
    float r_dot_wo = max(0.0, dot(R, wo));
    shine = (0.0 < exponent) ? pow(r_dot_wo, exponent) : 1.0;
  }
  return ambient * ambientColor +
         ns_dot_wi * diffuse * diffuseColor +
         shine * specular * specularColor;
}

void main(void)
{
  vec3 wo = normalize(varEyePos - varWorldPos);
  vec3 ns = normalize(varNormal);
  
  vec3 emissiveColor;
  vec3 ambientColor;
  vec3 diffuseColor;
  vec3 specularColor;
  float exponent;
  float alpha;
  
  if (gl_FrontFacing)
  {
    emissiveColor = frontEmissiveColor;
    ambientColor  = frontAmbientColor;
    diffuseColor  = frontDiffuseColor;
    specularColor = frontSpecularColor;
    exponent      = frontSpecularExponent;
    alpha         = frontOpacity;
  }
  else
  {
    ns = -ns;
    emissiveColor = backEmissiveColor;
    ambientColor  = backAmbientColor;
    diffuseColor  = backDiffuseColor;
    specularColor = backSpecularColor;
    exponent      = backSpecularExponent;
    alpha         = backOpacity;
  }

  if ( textureEnable )
  {
    vec4 texColor = texture2D(sampler, varTexCoord0);
    switch (envMode)
    {
      case TEM_REPLACE:
        diffuseColor = texColor.xyz;
        alpha = texColor.a;
        break;
      default:      
      case TEM_MODULATE:
        diffuseColor *= texColor.xyz;
        alpha *= texColor.a;
        break;
      case TEM_DECAL:
        diffuseColor = mix(diffuseColor, texColor.xyz, texColor.a);
        break;
    }
  }

  switch ( alphaFunction )
  {
    case AF_NEVER:                                         discard;            // Never draw the fragment
    case AF_LESS:     if ( alphaThreshold <= alpha ) { discard; } break;   // Draw the fragment if fragment.a <  threshold
    case AF_EQUAL:    if ( alphaThreshold != alpha ) { discard; } break;   // Draw the fragment if fragment.a == threshold
    case AF_LEQUAL:   if ( alphaThreshold <  alpha ) { discard; } break;   // Draw the fragment if fragment.a <= threshold
    case AF_GREATER:  if ( alpha <= alphaThreshold ) { discard; } break;   // Draw the fragment if fragment.a >  threshold
    case AF_NOTEQUAL: if ( alpha == alphaThreshold ) { discard; } break;   // Draw the fragment if fragment.a != threshold
    case AF_GEQUAL:   if ( alpha <  alphaThreshold ) { discard; } break;   // Draw the fragment if fragment.a >= threshold
    default:
    case AF_ALWAYS:                                                   break;   // Always draw the fragment
  }

  vec4 rgba = unlitColor;
  if ( lightingEnabled )
  {
    vec3 rgb = emissiveColor +
               ambientColor * sys_SceneAmbientLight;
    if ( gl_FrontFacing || twoSidedLighting )
    {
      vec3 ambient;
      vec3 diffuse;
      vec3 specular;
      vec3 wi;
      for (int i = 0; i < sys_NumLights; ++i)
      {
        sampleLight(sys_Lights[i], varWorldPos, wi, ambient, diffuse, specular);
        rgb += eval(wo, ns, wi, 
                   ambientColor, diffuseColor, specularColor, 
                   ambient, diffuse, specular, exponent);
      }
    }
    rgba = vec4( rgb, alpha );
  }
  emitColor( rgba );
}
