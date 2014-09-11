struct NVSGLight
{
  vec4    lightAmbient;                //!< Specifies the ambient RGBA intensity of the light.
  vec4    lightDiffuse;                //!< Specifies the diffuse RGBA intensity of the light.
  vec4    lightSpecular;               //!< Specifies the specular RGBA intensity of the light.
  vec4    lightPosition;               //!< Specifies the light position in world coordinates.
  vec4    lightDirection;              //!< Specifies the light direction in world coordinates.
  float   lightSpotExponent;           //!< Specifies the intensity distribution of the light.
  float   lightSpotCutoff;             //!< Specifies the maximum spread angle of the light.
  float   lightConstantAttenuation;    //!< Specifies the constant light attenuation factor.
  float   lightLinearAttenuation;      //!< Specifies the linear light attenuation factor.
  float   lightQuadraticAttenuation;   //!< Specifies the quadratic light attenuation factor.
};

#define MAXLIGHTS 128
layout(std140) uniform sys_LightsBuffer
{
  uniform vec3      sys_SceneAmbientLight;
  uniform int       sys_NumLights;
  uniform NVSGLight sys_Lights[MAXLIGHTS];
};

#ifndef PI
#define PI          3.14159265358979f
#define PI_HALF     ( 0.5f * PI )
#define PI_TWO      ( 2.0f * PI )
#define ONE_OVER_PI ( 1.0f / PI )
#define TWO_OVER_PI ( 2.0f / PI )
#define PI_SQUARE   ( PI * PI )
#endif

void sampleLight(in NVSGLight light,
                 in vec3 pos,
                 out vec3 wi,
                 out vec3 opAmbient,
                 out vec3 opDiffuse,
                 out vec3 opSpecular)
{
  float att = 1.0;
  if (light.lightPosition.w == 0.0f)  // directional light
  {
    wi = normalize(light.lightPosition.xyz);
  }
  else  // point or spot light
  {
    wi = light.lightPosition.xyz - pos;
    float dist = length(wi);
    wi = wi / dist;  // == normalize(wi);
    att = 1.0f / (light.lightConstantAttenuation + (light.lightLinearAttenuation + light.lightQuadraticAttenuation * dist) * dist);
    if (light.lightSpotCutoff < 180.0f) // spot light
    {
      float spot = max(0.0f, dot(wi, -light.lightDirection.xyz));
      att *= (spot >= cos(light.lightSpotCutoff * PI / 180.0f)) ? pow(spot, light.lightSpotExponent) : 0.0f;
    }
  }
  opAmbient  = att * light.lightAmbient.xyz;
  opDiffuse  = att * light.lightDiffuse.xyz;
  opSpecular = att * light.lightSpecular.xyz;
}

