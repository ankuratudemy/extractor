@description('Resource group location')
param rgLocation string

@description('VNET resource name')
param vnetName string

@description('VNET Address Space (CIDR notation, /23 or greater)')
param vnetAddressSpace string = '10.0.0.0/16'

@description('Subnet resource name')
param containerAppSubnetName string

@description('Subnet Address Prefix (CIDR notation, /23 or greater)')
param subnetAddressPrefix string = '10.0.0.0/21'

@description('Container Apps Environment resource name')
param cappEnvName string

@description('extractor image url')
param extractorContainerImage string

@description('extractor image Tag')
param extractorContainerTag string
@description('extractor server image url')
param extarctorServerContainerImage string
@description('extractor server image tag')
param extarctorServerContainerTag string

@description('Log Analytics resource name')
param cappLogAnalyticsName string

param acrClientId string

@secure()
param acrClientSecret string


// @description('Name of the connected Container Registry')
// param containerRegistryName string

// resource containerRegistry 'Microsoft.ContainerRegistry/registries@2021-12-01-preview' = {
//   name: containerRegistryName
//   location: rgLocation
//   sku: {
//     name: 'Basic'
//   }
//   properties: {
//     adminUserEnabled: true
//   }
// }


resource vnet 'Microsoft.Network/virtualNetworks@2023-04-01' = {
  name: vnetName
  location: rgLocation
  properties: {
    addressSpace: {
      addressPrefixes: [ vnetAddressSpace ]
    }
    subnets: [
      {
        name: containerAppSubnetName
        properties: {
          addressPrefix: subnetAddressPrefix
          serviceEndpoints: [
            {
              service: 'Microsoft.Storage'
              locations: [ rgLocation ]
            }
          ]
        }
      }
    ]
  }
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2021-06-01' = {
  name: cappLogAnalyticsName
  location: rgLocation
  properties: {
    sku: { name: 'PerGB2018' }
  }
}

resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2022-03-01' = {
  location: rgLocation
  name: cappEnvName
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    vnetConfiguration: {
      infrastructureSubnetId: vnet.properties.subnets[0].id
      dockerBridgeCidr: '10.2.0.1/16'
      platformReservedCidr: '10.1.0.0/16'
      platformReservedDnsIP: '10.1.0.2'
      // runtimeSubnetId: resourceId('Microsoft.Network/virtualNetworks/subnets', vnetName, subnetName)
    }
  }
}

resource extractorServer 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'extractor-server'
  location: rgLocation
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnvironment.id
    configuration: {
      secrets: [
        {
          name: 'container-registry-password'
          value: acrClientSecret
        }
      ]
      ingress: {
        external: false
        targetPort: 9998
        allowInsecure: true
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
        stickySessions: {
          affinity:'none'
        }
      }
      registries: [
        {
          server: 'extractor.azurecr.io'
          username: acrClientId
          passwordSecretRef: 'container-registry-password'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'extractor-server'
          image: '${extarctorServerContainerImage}:${extarctorServerContainerTag}'
          resources: {
            cpu: '1'
            memory: '2Gi'
          }
          probes: [
            {
              failureThreshold: 2
              httpGet: {
                // host: 'string'
                // httpHeaders: [
                //   {
                //     name: 'string'
                //     value: 'string'
                //   }
                // ]
                path: '/tika'
                port: 9998
                scheme: 'HTTP'
              }
              initialDelaySeconds: 2
              periodSeconds: 30
              successThreshold: 1
              // tcpSocket: {
              //   host: 'string'
              //   port: int
              // }
              // terminationGracePeriodSeconds: 30 // fixed at 30 for now
              timeoutSeconds: 2
              type: 'Liveness'
            }
          ]
        }
      ]
      scale: {
        maxReplicas: 50
        minReplicas: 0
        rules: [
          {
            name: 'http-scale-rule'
            http: {
              metadata: {
                concurrentRequests: '1'
              }
            }
          }
        ]
      }
    }
  }
}

resource extractor 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'extractor'
  location: rgLocation
  identity: {
    type: 'SystemAssigned'
   
  }
  properties: {
    managedEnvironmentId: containerAppEnvironment.id
    configuration: {
      secrets: [
        {
          name: 'container-registry-password'
          value: acrClientSecret
        }
      ]
      registries: [
        {
          server: 'extractor.azurecr.io'
          username: acrClientId
          passwordSecretRef: 'container-registry-password'
        }
      ]
      ingress: {
        external: true
        targetPort: 5000
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
    }
    template: {
      containers: [
        {
          name: 'extractor'
          image: '${extractorContainerImage}:${extractorContainerTag}'
          resources: {
            cpu: '2'
            memory: '4Gi'
          }
          env: [
            {
              name: 'SERVER_URL'
              value: extractorServer.name
            }
          ]
          probes: [
            {
              failureThreshold: 2
              httpGet: {
                // host: 'string'
                // httpHeaders: [
                //   {
                //     name: 'string'
                //     value: 'string'
                //   }
                // ]
                path: '/health'
                port: 5000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 2
              periodSeconds: 10
              successThreshold: 1
              // tcpSocket: {
              //   host: 'string'
              //   port: int
              // }
              // terminationGracePeriodSeconds: 30 // fixed at 30 for now
              timeoutSeconds: 2
              type: 'Liveness'
            }
          ]
        }
      ]
      scale: {
        maxReplicas: 25
        minReplicas: 0
        rules: [
          {
            name: 'http-scale-rule'
            http: {
              metadata: {
                concurrentRequests: '1'
              }
            }
          }
        ]
      }
    }
  }
}

output vnetName string = vnetName
output location string = rgLocation
output cappEnvName string = cappEnvName
