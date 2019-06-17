// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::result;

use futures::sync::oneshot;
use hyper::Method;
use serde_json::Value;

use request::{IntoParsedRequest, ParsedRequest};
use vmm::VmmAction;

// The names of the members from this enum must precisely correspond (as a string) to the possible
// values of "action_type" from the json request body. This is useful to get a strongly typed
// struct from the Serde deserialization process.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
enum ActionType {
    BlockDeviceRescan,
    FlushMetrics,
    InstanceStart,
    PauseToSnapshot,
    PauseVCPUs,
    ResumeFromSnapshot,
    ResumeVCPUs,
    SendCtrlAltDel,
}

// The model of the json body from a sync request. We use Serde to transform each associated
// json body into this.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActionBody {
    action_type: ActionType,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<Value>,
}

fn validate_payload(action_body: &ActionBody) -> Result<(), String> {
    match action_body.action_type {
        ActionType::BlockDeviceRescan => {
            match action_body.payload {
                Some(ref payload) => {
                    // Expecting to have drive_id as a String in the payload.
                    if !payload.is_string() {
                        return Err(
                            "Invalid payload type. Expected a string representing the drive_id"
                                .to_string(),
                        );
                    }
                    Ok(())
                }
                None => Err("Payload is required for block device rescan.".to_string()),
            }
        }
        ActionType::FlushMetrics
        | ActionType::InstanceStart
        | ActionType::SendCtrlAltDel
        | ActionType::PauseVCPUs
        | ActionType::ResumeVCPUs
        | ActionType::PauseToSnapshot
        | ActionType::ResumeFromSnapshot => {
            // These actions don't have a payload.
            if action_body.payload.is_some() {
                return Err(format!(
                    "{:?} does not support a payload.",
                    action_body.action_type
                ));
            }
            Ok(())
        }
    }
}

impl IntoParsedRequest for ActionBody {
    fn into_parsed_request(
        self,
        _: Option<String>,
        _: Method,
    ) -> result::Result<ParsedRequest, String> {
        validate_payload(&self)?;
        match self.action_type {
            ActionType::BlockDeviceRescan => {
                // Safe to unwrap because we validated the payload in the validate_payload func.
                let block_device_id = self.payload.unwrap().as_str().unwrap().to_string();
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmAction::RescanBlockDevice(block_device_id, sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::FlushMetrics => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmAction::FlushMetrics(sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::InstanceStart => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmAction::StartMicroVm(sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::PauseToSnapshot => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmAction::PauseToSnapshot(sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::PauseVCPUs => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmAction::PauseVCPUs(sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::ResumeFromSnapshot => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmAction::ResumeFromSnapshot(sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::ResumeVCPUs => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmAction::ResumeVCPUs(sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::SendCtrlAltDel => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmAction::SendCtrlAltDel(sync_sender),
                    sync_receiver,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ActionType::*;
    use super::*;
    use serde_json;

    fn action_body_no_payload(action_type: ActionType) -> ActionBody {
        ActionBody {
            action_type,
            payload: None,
        }
    }

    fn action_body_dummy_payload(action_type: ActionType) -> ActionBody {
        ActionBody {
            action_type,
            payload: Some(Value::String("dummy-payload".to_string())),
        }
    }

    #[test]
    fn test_validate_payload() {
        // Test BlockDeviceRescan.
        assert!(validate_payload(&action_body_dummy_payload(BlockDeviceRescan)).is_ok());
        // Error case: no payload.
        assert!(validate_payload(&action_body_no_payload(BlockDeviceRescan)).is_err());
        // Error case: payload is not String.
        let action_body = ActionBody {
            action_type: ActionType::BlockDeviceRescan,
            payload: Some(Value::Bool(false)),
        };
        assert!(validate_payload(&action_body).is_err());

        // Test FlushMetrics.
        assert!(validate_payload(&action_body_no_payload(FlushMetrics)).is_ok());
        // Error case: FlushMetrics with payload.
        let res = validate_payload(&action_body_dummy_payload(FlushMetrics));
        assert!(res.is_err());
        // Also test the error formatting.
        assert_eq!(res.unwrap_err(), "FlushMetrics does not support a payload.");

        // Test InstanceStart.
        assert!(validate_payload(&action_body_no_payload(InstanceStart)).is_ok());
        // Error case: InstanceStart with payload.
        assert!(validate_payload(&action_body_dummy_payload(InstanceStart)).is_err());

        // Test PauseToSnapshot.
        assert!(validate_payload(&action_body_no_payload(PauseToSnapshot)).is_ok());
        // Error case: PauseToSnapshot with payload.
        assert!(validate_payload(&action_body_dummy_payload(PauseToSnapshot)).is_err());

        // Test PauseVCPUs.
        assert!(validate_payload(&action_body_no_payload(PauseVCPUs)).is_ok());
        // Error case: PauseVCPUs with payload.
        assert!(validate_payload(&action_body_dummy_payload(PauseVCPUs)).is_err());

        // Test ResumeFromSnapshot.
        assert!(validate_payload(&action_body_no_payload(ResumeFromSnapshot)).is_ok());
        // Error case: ResumeFromSnapshot with payload.
        assert!(validate_payload(&action_body_dummy_payload(ResumeFromSnapshot)).is_err());

        // Test ResumeVCPUs.
        assert!(validate_payload(&action_body_no_payload(ResumeVCPUs)).is_ok());
        // Error case: ResumeVCPUs with payload.
        assert!(validate_payload(&action_body_dummy_payload(ResumeVCPUs)).is_err());

        // Test SendCtrlAltDel.
        assert!(validate_payload(&action_body_no_payload(SendCtrlAltDel)).is_ok());
        // Error case: SendCtrlAltDel with payload.
        assert!(validate_payload(&action_body_dummy_payload(SendCtrlAltDel)).is_err());
    }

    #[test]
    fn test_into_parsed_request() {
        // Test BlockDeviceRescan.
        {
            let json = r#"{
                "action_type": "BlockDeviceRescan",
                "payload": "dummy_id"
              }"#;
            let (sender, receiver) = oneshot::channel();
            let req = ParsedRequest::Sync(
                VmmAction::RescanBlockDevice("dummy_id".to_string(), sender),
                receiver,
            );
            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));
        }

        // Test FlushMetrics.
        {
            let json = r#"{
                "action_type": "FlushMetrics"
            }"#;

            let (sender, receiver) = oneshot::channel();
            let req: ParsedRequest = ParsedRequest::Sync(VmmAction::FlushMetrics(sender), receiver);
            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));

            let json = r#"{
                "action_type": "FlushMetrics",
                "payload": "metrics-payload"
            }"#;

            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            let res = result.unwrap().into_parsed_request(None, Method::Put);
            assert!(res.is_err());
            assert!(res == Err("FlushMetrics does not support a payload.".to_string()));
        }

        // Test InstanceStart.
        {
            let json = r#"{
                "action_type": "InstanceStart"
            }"#;

            let (sender, receiver) = oneshot::channel();
            let req: ParsedRequest = ParsedRequest::Sync(VmmAction::StartMicroVm(sender), receiver);
            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));
        }

        // Test PauseToSnapshot.
        {
            let json = r#"{
                "action_type": "PauseToSnapshot"
            }"#;

            let (sender, receiver) = oneshot::channel();
            let req: ParsedRequest =
                ParsedRequest::Sync(VmmAction::PauseToSnapshot(sender), receiver);
            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));
        }

        // Test PauseVCPUs.
        {
            let json = r#"{
                "action_type": "PauseVCPUs"
            }"#;

            let (sender, receiver) = oneshot::channel();
            let req: ParsedRequest = ParsedRequest::Sync(VmmAction::PauseVCPUs(sender), receiver);
            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));
        }

        // Test ResumeFromSnapshot.
        {
            let json = r#"{
                "action_type": "ResumeFromSnapshot"
            }"#;

            let (sender, receiver) = oneshot::channel();
            let req: ParsedRequest =
                ParsedRequest::Sync(VmmAction::ResumeFromSnapshot(sender), receiver);
            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));
        }

        // Test ResumeVCPUs.
        {
            let json = r#"{
                "action_type": "ResumeVCPUs"
            }"#;

            let (sender, receiver) = oneshot::channel();
            let req: ParsedRequest = ParsedRequest::Sync(VmmAction::ResumeVCPUs(sender), receiver);
            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));
        }

        // Test SendCtrlAltDel.
        {
            let json = r#"{
                "action_type": "SendCtrlAltDel"
            }"#;

            let (sender, receiver) = oneshot::channel();
            let req: ParsedRequest =
                ParsedRequest::Sync(VmmAction::SendCtrlAltDel(sender), receiver);
            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));
        }
    }
}
