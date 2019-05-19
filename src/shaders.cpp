#define STRINGIFY(A) #A

// vertex shader
const char *particleVS = STRINGIFY(
                               uniform float pointRadius;  // point size in world space
                               uniform float pointScale;   // scale to calculate size in pixels
                               uniform float densityScale;
                               uniform float densityOffset;
                               void main()
{
    vec4 wpos = vec4(gl_Vertex.xyz, 1.0);
            gl_Position = gl_ModelViewProjectionMatrix *wpos;

    vec4 eyeSpacePos = gl_ModelViewMatrix *wpos;
    float dist = length(eyeSpacePos.xyz);
            gl_PointSize = pointRadius * (pointScale / dist);

            gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_TexCoord[1] = gl_Vertex.w;

            gl_FrontColor = gl_Color;

//    // calculate window-space point size
//    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
//    float dist = length(posEye);
//    gl_PointSize = pointRadius * (pointScale / dist);
//
//    gl_TexCoord[0] = gl_MultiTexCoord0;
//    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
//
//    gl_FrontColor = gl_Color;
}
                           );

// pixel shader for rendering points as shaded spheres
const char *particlePS = STRINGIFY(
                                    void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > gl_TexCoord[1].w) discard;   // kill pixels outside circle

    N.z = sqrt(1.0-mag);

    float alpha = gl_TexCoord[1].w;
    gl_FragColor = vec4(gl_Color.xyz * alpha, alpha);
}
                                );


