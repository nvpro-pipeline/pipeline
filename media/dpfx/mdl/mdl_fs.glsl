
void main(void)
{
  stateNormal = normalize( varNormal );
  if ( ! gl_FrontFacing )
  {
    stateNormal = - stateNormal;
  }

  texCoord0 = varTexCoord0;
  tangent   = normalize( varTangent );
  binormal  = normalize( varBinormal );
  viewDir   = normalize( varEyePos - varWorldPos );

  vec4 rgba = vec4( 0.0f, 0.0f, 0.0f, 0.0f );
  evalTemporaries(stateNormal);
  float cutoutOpacity = evalCutoutOpacity(stateNormal);
  if (0.0f < cutoutOpacity)
  {
    vec3 normal = evalNormal( stateNormal );
    materialIOR = evalIOR(normal);

    vec3 materialEmissive = vec3( 0.0f, 0.0f, 0.0f );
    bool useFront = gl_FrontFacing;
    if ( useFront )
    {
      materialEmissive = evalMaterialEmissiveFront(normal);
    }
    else
    {
      // there's no emission on the back-side, unless thinWalled is true
      useFront = !evalThinWalled();
      if ( !useFront )
      {
        materialEmissive = evalMaterialEmissiveBack(normal);
      }
      materialIOR = 1.0f / materialIOR;
    }

    rgba = vec4( materialEmissive, 0.0f );
    if ( 0 < sys_NumLights )
    {
      vec3 lightAmbient;
      for ( int i=0 ; i<sys_NumLights ; i++ )
      {
        sampleLight( sys_Lights[i], varWorldPos, lightDir, lightAmbient, lightDiffuse, lightSpecular);
        evalTemporariesPerLightSource();
        rgba += useFront ? evalColorFront(normal) : evalColorBack(normal);
      }
      rgba.a /= sys_NumLights;
    }
    else
    {
      rgba.a = 1.0f;
    }
    rgba.a *= cutoutOpacity;

    if ( 0.0f < rgba.a )
    {
      lightDir = reflect(-viewDir, normal);
      rgba.rgb += (useFront ? evalEnvironmentFront(normal) : evalEnvironmentBack(normal)).rgb;
    }
  }
  emitColor( rgba );
}
