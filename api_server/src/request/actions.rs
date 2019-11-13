// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::result;

use futures::sync::oneshot;
use hyper::Method;
use serde_json::Value;

use super::{VmmAction, VmmRequest};
use request::{IntoParsedRequest, ParsedRequest};

// The names of the members from this enum must precisely correspond (as a string) to the possible
// values of "action_type" from the json request body. This is useful to get a strongly typed
// struct from the Serde deserialization process.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
enum ActionType {
    BlockDeviceRescan,
    FlushMetrics,
    InstanceStart,
    PauseVCPUs,
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
                    // Expecting to have a String in the payload.
                    if !payload.is_string() {
                        return Err(
                            "Invalid payload type. Expected a string representing the drive ID"
                                .to_string(),
                        );
                    }
                    Ok(())
                }
                None => Err(format!(
                    "Payload is required for {:?}.",
                    action_body.action_type
                )),
            }
        }
        ActionType::FlushMetrics
        | ActionType::InstanceStart
        | ActionType::SendCtrlAltDel
        | ActionType::PauseVCPUs
        | ActionType::ResumeVCPUs => {
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
                    VmmRequest::new(VmmAction::RescanBlockDevice(block_device_id), sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::FlushMetrics => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmRequest::new(VmmAction::FlushMetrics, sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::InstanceStart => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmRequest::new(VmmAction::StartMicroVm, sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::PauseVCPUs => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmRequest::new(VmmAction::PauseVCPUs, sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::ResumeVCPUs => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmRequest::new(VmmAction::ResumeVCPUs, sync_sender),
                    sync_receiver,
                ))
            }
            ActionType::SendCtrlAltDel => {
                let (sync_sender, sync_receiver) = oneshot::channel();
                Ok(ParsedRequest::Sync(
                    VmmRequest::new(VmmAction::SendCtrlAltDel, sync_sender),
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

    fn action_body_string_payload(action_type: ActionType) -> ActionBody {
        ActionBody {
            action_type,
            payload: Some(Value::String("dummy-payload".to_string())),
        }
    }

    fn action_body_bool_payload(action_type: ActionType) -> ActionBody {
        ActionBody {
            action_type,
            payload: Some(Value::Bool(false)),
        }
    }

    #[test]
    fn test_validate_payload() {
        // Test BlockDeviceRescan.
        assert!(validate_payload(&action_body_string_payload(BlockDeviceRescan)).is_ok());
        // Error case: no payload.
        assert!(validate_payload(&action_body_no_payload(BlockDeviceRescan)).is_err());
        // Error case: payload is not String.
        assert!(validate_payload(&action_body_bool_payload(BlockDeviceRescan)).is_err());

        // Test FlushMetrics.
        assert!(validate_payload(&action_body_no_payload(FlushMetrics)).is_ok());
        // Error case: FlushMetrics with payload.
        assert!(validate_payload(&action_body_string_payload(FlushMetrics)).is_err());

        // Test InstanceStart.
        assert!(validate_payload(&action_body_no_payload(InstanceStart)).is_ok());
        // Error case: InstanceStart with payload.
        assert!(validate_payload(&action_body_string_payload(InstanceStart)).is_err());

        // Test PauseVCPUs.
        assert!(validate_payload(&action_body_no_payload(PauseVCPUs)).is_ok());
        // Error case: PauseVCPUs with payload.
        assert!(validate_payload(&action_body_string_payload(PauseVCPUs)).is_err());

        // Test ResumeVCPUs.
        assert!(validate_payload(&action_body_no_payload(ResumeVCPUs)).is_ok());
        // Error case: ResumeVCPUs with payload.
        assert!(validate_payload(&action_body_string_payload(ResumeVCPUs)).is_err());

        // Test SendCtrlAltDel.
        assert!(validate_payload(&action_body_no_payload(SendCtrlAltDel)).is_ok());
        // Error case: SendCtrlAltDel with payload.
        assert!(validate_payload(&action_body_string_payload(SendCtrlAltDel)).is_err());
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
                VmmRequest::new(VmmAction::RescanBlockDevice("dummy_id".to_string()), sender),
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
            let req: ParsedRequest =
                ParsedRequest::Sync(VmmRequest::new(VmmAction::FlushMetrics, sender), receiver);
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
            let req: ParsedRequest =
                ParsedRequest::Sync(VmmRequest::new(VmmAction::StartMicroVm, sender), receiver);
            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            assert!(result
                .unwrap()
                .into_parsed_request(None, Method::Put)
                .unwrap()
                .eq(&req));

            let json = r#"{
                "action_type": "InstanceStart",
                "payload": "dummy-payload"
            }"#;

            let result: Result<ActionBody, serde_json::Error> = serde_json::from_str(json);
            assert!(result.is_ok());
            let res = result.unwrap().into_parsed_request(None, Method::Put);
            assert!(res.is_err());
            assert!(res == Err("InstanceStart does not support a payload.".to_string()));
        }

        // Test PauseVCPUs.
        {
            let json = r#"{
                "action_type": "PauseVCPUs"
            }"#;

            let (sender, receiver) = oneshot::channel();
            let req: ParsedRequest =
                ParsedRequest::Sync(VmmRequest::new(VmmAction::PauseVCPUs, sender), receiver);
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
            let req: ParsedRequest =
                ParsedRequest::Sync(VmmRequest::new(VmmAction::ResumeVCPUs, sender), receiver);
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
                ParsedRequest::Sync(VmmRequest::new(VmmAction::SendCtrlAltDel, sender), receiver);
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
