float luminance( in vec3 v )
{
  // Luma coefficients according to ITU-R Recommendation BT.709 (http://en.wikipedia.org/wiki/Rec._709)
  return( 0.2126f * v.r + 0.7152f * v.g + 0.0722f * v.b );
}

void main(void)
{
  vec4 materialEmission = evaluateColor( emissionColor, emissionSampler, emissionTC );

  vec3 rgb = materialEmission.xyz;

  if ( LM_CONSTANT < lighting )
  {
    vec3 wo = normalize(varEyePos - varWorldPos);
    vec3 ns = bumpmap( normalize(varNormal) );

    vec4 materialAmbient  = evaluateColor( ambientColor, ambientSampler, ambientTC );
    vec4 materialDiffuse  = evaluateColor( diffuseColor, diffuseSampler, diffuseTC );
    vec4 materialSpecular = evaluateColor( specularColor, specularSampler, specularTC );
    // vec4 materialReflective = evaluateColor( reflectiveColor, reflectiveSampler, reflectiveTC ); // When using the sammpler specified inside the COLLADA material.

    vec3 materialReflective = vec3(0.0f); 
    if (sys_EnvironmentSamplerEnabled)
    {
      vec3 r  = reflect(-wo, ns);
      vec2 tc = vec2( ( atan( r.x, -r.z ) + PI ) / ( 2.0f * PI ), acos( -r.y ) / PI );
      materialReflective = texture(sys_EnvironmentSampler, tc).xyz * reflectiveColor.xyz * reflectivity;
    }

    rgb += materialAmbient.xyz * sys_SceneAmbientLight + materialReflective;

    for ( int i = 0; i < sys_NumLights; ++i )
    {
      vec3 wi;
      vec3 lightAmbient;
      vec3 lightDiffuse;
      vec3 lightSpecular;
    
      sampleLight( sys_Lights[i], varWorldPos, wi, lightAmbient, lightDiffuse, lightSpecular );

      float ns_dot_wi = max( 0.0f, dot( ns, wi ) );
      if ( (LM_LAMBERT < lighting) && (0.0f < ns_dot_wi) )
      {
        float shine = 1.0f;
        if ( 0.0f < shininess )
        {
          float cosine;
          if ( LM_BLINN < lighting) 
          {
            vec3 R = reflect( -wi, ns );  // Phong
            cosine = dot( R, wo );
          }
          else
          {
            vec3 H = normalize( wo + wi );
            cosine = dot( H, ns );
          }
          shine = pow( max( 0.0f, cosine), shininess );
        }
        rgb += shine * lightSpecular * materialSpecular.xyz;
      }
      rgb += lightAmbient * materialAmbient.xyz
           + ns_dot_wi * lightDiffuse * materialDiffuse.xyz;
    }
  }
  
  vec4 materialTransparent = evaluateColor( transparentColor, transparentSampler, transparentTC );
  float a = ( rgbTransparency ? 1.0f - luminance( materialTransparent.rgb ) : materialTransparent.a ) * transparency;
  
  emitColor( vec4( rgb, a ) );
}
